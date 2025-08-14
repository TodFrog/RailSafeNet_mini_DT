import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
from tqdm import tqdm
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import AdamW

# ✅ [핵심] wandb 시각화를 위한 클래스 및 색상 맵 (기존 코드와 동일)
RAILSEM_COLOR_MAP = [
    [0, 0, 0], [111, 74, 0], [70, 70, 70], [128, 64, 128], [244, 35, 232],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [107, 142, 35], [152, 251, 152],
    [220, 220, 0], [250, 170, 30], [102, 102, 156], [190, 153, 153]
]
CLASS_ID_TO_NAME = {
    0: "background", 1: "rail-track", 2: "rail-bed", 3: "rail-platform", 4: "rail-sign",
    5: "rail-pole", 6: "rail-traffic-light", 7: "vehicle-car", 8: "vehicle-truck", 9: "vehicle-tram",
    10: "vehicle-train", 11: "person", 12: "animal", 13: "vegetation", 14: "terrain",
    15: "sky", 16: "building", 17: "infrastructure", 18: "object"
}

# --- 경로 설정 (사용자 환경에 맞게 수정) ---
PATH_JPGS = "/home/mmc-server3/Server/Datasets/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server3/Server/Datasets/rs19_val/uint8/rs19_val"
# ⚠️ Fine-tuning에 사용할 B3 모델 .pth 파일 경로
PATH_MODEL_PTH = "/home/mmc-server3/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"
# Fine-tuning된 모델을 저장할 경로
PATH_MODELS = "RailNet_DT/models_finetuned"
NUM_CLASSES = 19
WANDB_LOGGING = True # wandb 사용 여부

# ✅ [핵심] DDP 설정 및 해제 함수 (기존 코드와 동일)
def ddp_setup():
    dist.init_process_group("nccl")
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

# finetuning_DDP.py 스크립트의 create_b3_for_finetune 함수를 교체하세요.

def create_b3_for_finetune(pth_path, num_classes):
    """
    .pth 파일에서 가중치를 읽어 fine-tuning을 위한 SegFormer-B3 모델을 생성합니다.
    """
    # 1. 최종 목표 모델 생성 (B3 백본 + 19 클래스 헤드)
    print("Step 1: Creating the target model (B3 backbone + 19-class head)...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # 2. 모델 '객체'를 통째로 로드 (weights_only=False 사용)
    print(f"Step 2: Loading model object from {pth_path} with weights_only=False...")
    loaded_object = torch.load(pth_path, map_location="cpu", weights_only=False)

    # 3. DataParallel 래퍼 확인 및 실제 모델 추출
    if isinstance(loaded_object, nn.DataParallel):
        print("Step 3: DataParallel wrapper detected. Extracting the inner model...")
        source_model = loaded_object.module
    else:
        print("Step 3: No DataParallel wrapper detected. Using the loaded object directly...")
        source_model = loaded_object
        
    # 4. 실제 모델에서 state_dict(가중치) 추출
    print("Step 4: Extracting the state_dict from the source model...")
    state_dict = source_model.state_dict()

    # ✅ [핵심 수정] 'decode_head'와 관련된 모든 키를 state_dict에서 제거합니다.
    # 이제 이 딕셔너리에는 백본 가중치만 남게 됩니다.
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('decode_head.')}
    
    # 5. 백본 가중치만 담긴 딕셔너리를 로드합니다.
    #    이제 불일치하는 부분이 아예 없으므로, strict=False가 확실하게 동작합니다.
    print("Step 5: Injecting BACKBONE weights into the new model...")
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print("Model successfully prepared for fine-tuning!")
    return model


# ✅ [핵심] 백본 가중치 동결 함수
def freeze_backbone(model):
    """
    모델의 백본(segformer) 부분의 가중치 학습을 막습니다(동결).
    decode_head 부분만 학습됩니다.
    """
    print("Freezing backbone weights. Only the decode_head will be trained.")
    for name, param in model.named_parameters():
        if 'decode_head' not in name:
            param.requires_grad = False

def main():
    ddp_setup()
    # local_rank뿐만 아니라 rank와 world_size도 가져옵니다.
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    config = {
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 4,
        'optimizer': 'adamw'
    }

    # ✅ [핵심] rank 0 프로세스만 wandb를 초기화하고 폴더를 생성합니다.
    if rank == 0:
        if WANDB_LOGGING:
            wandb.init(project="FineTune_SegFormer", config=config, name=f"finetune_b3_bs{config['batch_size']}_lr{config['learning_rate']}")
        if not os.path.exists(PATH_MODELS):
            os.makedirs(PATH_MODELS)
    
    per_gpu_batch_size = config['batch_size'] // world_size
    
    # --- 모델 생성 및 로드 동기화 ---
    # ✅ [핵심] rank 0이 아닌 다른 프로세스들은 여기서 대기합니다.
    if rank != 0:
        dist.barrier()

    # rank 0이 먼저 모델 생성 및 파일 로드를 안전하게 수행합니다.
    model = create_b3_for_finetune(PATH_MODEL_PTH, NUM_CLASSES)
    freeze_backbone(model)

    # ✅ [핵심] rank 0이 준비를 마치면, 다른 프로세스들이 대기를 해제하고 진행하도록 신호를 줍니다.
    if rank == 0:
        dist.barrier()
    # --------------------------------

    # 이제 모든 프로세스가 각자의 GPU에 모델을 올리고 DDP로 감쌉니다.
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0 and WANDB_LOGGING:
        wandb.watch(model, log="all", log_freq=100)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config['epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ... 이하 데이터 로더 및 학습 루프는 기존 코드와 동일합니다 ...
    image_processor = SegformerImageProcessor(size={"height": 1024, "width": 1024})
    
    train_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Train', NUM_CLASSES)
    valid_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Valid', NUM_CLASSES)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=valid_sampler, num_workers=4)
    
    best_miou = 0.0

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        # filter(lambda p: p.requires_grad, model.parameters()) 를 통해 학습되는 파라미터만 학습하도록 설정
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate']

        total_loss = 0
        
        loader_tqdm = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}", disable=(rank != 0))
        for inputs, masks in loader_tqdm:
            inputs, masks = inputs.to(local_rank), masks.to(local_rank)
            optimizer.zero_grad()
            outputs = model(pixel_values=inputs).logits
            logits = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    if rank == 0 and WANDB_LOGGING:
        wandb.finish()
        
    ddp_cleanup()

if __name__ == "__main__":
    main()