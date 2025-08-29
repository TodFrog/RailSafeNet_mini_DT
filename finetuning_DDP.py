import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
from tqdm import tqdm
from collections import OrderedDict

# DDP 관련 import
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# AMP(Automatic Mixed Precision) 관련 import
from torch.cuda.amp import GradScaler, autocast

# --- 사용자 정의 모듈 및 transformers ---
from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.optim import AdamW

# --- 경로 설정 ---
PATH_JPGS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
PATH_MODEL_PTH = "/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"
PATH_MODELS = "/home/mmc-server4/RailSafeNet_mini_DT/models_finetuned_13cls"

# ✅ 1. NUM_CLASSES를 13으로 설정
NUM_CLASSES = 13
WANDB_LOGGING = True

# ✅ 2. 클래스 리매핑 설정
# 학습에 사용할 19개 클래스 중 13개의 원본 ID 리스트
CLASSES_TO_TRAIN = [0, 1, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16]

# 리매핑 텐서 생성 (GPU 연산을 위해 미리 생성)
# key: 원본 클래스 ID, value: 새로운 클래스 ID (0~12) 또는 255 (무시)
remapping_tensor = torch.full((19,), 255, dtype=torch.long)
for new_id, old_id in enumerate(CLASSES_TO_TRAIN):
    remapping_tensor[old_id] = new_id

# --- DDP 설정 및 해제 함수 ---
def ddp_setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def ddp_cleanup():
    dist.destroy_process_group()
    
# --- Fine-tuning을 위한 모델 준비 함수 (수정됨) ---
def prepare_model_for_finetuning(pth_path, num_classes):
    """
    새로운 B3 모델 아키텍처를 생성하고, pth 파일에서 백본 가중치만 로드합니다.
    """
    print("Step 1: Creating a fresh SegFormer B3 model for fine-tuning...")
    # 새로운 모델 아키텍처 생성 (13개 클래스용 classifier 포함)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        num_labels=num_classes,
        ignore_mismatched_sizes=True 
    )

    print(f"Step 2: Loading pretrained weights from {pth_path}...")
    try:
        # CPU로 체크포인트 로드
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
        
        # state_dict 추출 (DataParallel 또는 DDP 래퍼 핸들링)
        if isinstance(checkpoint, torch.nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
        
        # 'module.' 접두사 제거
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            cleaned_state_dict[name] = v
        
        # Classifier 가중치는 제외하고 백본 가중치만 로드
        print("Step 3: Loading BACKBONE weights into the new model...")
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        print("✅ Model backbone successfully initialized for fine-tuning!")
        return model

    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        raise

# --- 백본 동결 함수 ---
def freeze_backbone(model):
    print("Freezing backbone weights. Only the decode_head will be trained.")
    for name, param in model.named_parameters():
        if 'decode_head' not in name:
            param.requires_grad = False

# --- MIoU 계산 함수 ---
def get_miou(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    # union이 0인 경우를 대비하여 1e-6 더하기
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    return np.nanmean(iou)

# --- 학습률 웜업 스케줄러 클래스 ---
class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]
    
# --- 메인 실행 함수 ---
def main():
    ddp_setup()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    config = {
        'epochs': 50, 
        'learning_rate': 1e-4,
        'batch_size': 8, 
        'optimizer': 'adamw', 
        'grad_clip': 1.0,
        'warmup_steps': 500 
    }

    if rank == 0 and WANDB_LOGGING:
        wandb.init(project="FineTune_SegFormer_13cls_Stable", config=config, name=f"finetune_b3_warmup_lr{config['learning_rate']}")
        if not os.path.exists(PATH_MODELS): os.makedirs(PATH_MODELS)
    
    per_gpu_batch_size = config['batch_size'] // world_size
    
    # 모든 프로세스가 모델을 로드하기 전에 동기화
    if rank != 0: dist.barrier()
    model = prepare_model_for_finetuning(PATH_MODEL_PTH, NUM_CLASSES)
    freeze_backbone(model) 
    if rank == 0: dist.barrier()

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if rank == 0 and WANDB_LOGGING: wandb.watch(model, log="all", log_freq=100)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    
    main_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config['epochs'])
    warmup_scheduler = WarmupLR(optimizer, warmup_steps=config['warmup_steps'])

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ✅ 표준화된 데이터로더 사용
    image_processor = SegformerImageProcessor(size={"height": 1024, "width": 1024})
    train_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Train', val_fraction=0.1, num_labels=19)
    valid_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Valid', val_fraction=0.1, num_labels=19)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=train_sampler, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=valid_sampler, num_workers=4, pin_memory=True)
    
    scaler = GradScaler()
    best_miou = 0.0
    global_step = 0

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        loader_tqdm = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}", disable=(rank != 0))
        for i, (inputs, masks) in enumerate(loader_tqdm):
            inputs, masks = inputs.to(local_rank), masks.to(local_rank)

            # ✅ 데이터로더에서 받은 마스크를 13개 클래스로 리매핑
            masks = remapping_tensor.to(local_rank)[masks]
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(pixel_values=inputs).logits
                logits = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, step {i}. Skipping backward pass.")
                continue

            # ✅ 중복 backward() 호출 수정
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])

            scaler.step(optimizer)
            scaler.update()

            if global_step < config['warmup_steps']:
                warmup_scheduler.step()
            
            global_step += 1
            total_loss += loss.item()

        if epoch * len(train_loader) >= config['warmup_steps']:
             main_scheduler.step()

        model.eval()
        local_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for inputs, masks in valid_loader:
                inputs, masks = inputs.to(local_rank), masks.to(local_rank)
                
                # ✅ 검증 루프에도 동일한 리매핑 적용
                masks = remapping_tensor.to(local_rank)[masks]
                
                with autocast():
                    outputs = model(pixel_values=inputs).logits
                    logits = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                gts = masks.cpu().numpy()
                
                mask = (gts != 255) 
                hist = np.bincount(NUM_CLASSES * gts[mask].astype(int) + preds[mask], minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
                local_confusion_matrix += hist

        dist.all_reduce(torch.from_numpy(local_confusion_matrix).to(local_rank), op=dist.ReduceOp.SUM)
        
        if rank == 0:
            global_confusion_matrix = local_confusion_matrix
            final_miou = get_miou(global_confusion_matrix)
            avg_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{config['epochs']}: Train loss: {avg_loss:.4f} | Val MIoU: {final_miou:.4f} | lr: {current_lr:.6f}")
            
            if WANDB_LOGGING:
                wandb.log({"train_loss": avg_loss, "val_MIoU": final_miou, "lr": current_lr, "epoch": epoch})

            if final_miou > best_miou:
                best_miou = final_miou
                save_path = os.path.join(PATH_MODELS, f'finetuned_b3_13cls_best_model.pth')
                # DDP 래퍼가 아닌 실제 모델의 state_dict 저장
                torch.save(model.module.state_dict(), save_path)
                print(f"Epoch {epoch+1}: Best model saved with MIoU: {best_miou:.4f}")
        
        dist.barrier()
    
    ddp_cleanup()

if __name__ == "__main__":
    main()
