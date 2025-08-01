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
import copy
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import Adam, Adagrad, AdamW

# ✅ [기능 복원] 클래스별 색상 맵 및 이름 정의 (RailSem19 데이터셋 기준)
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

def ddp_setup():
    dist.init_process_group("nccl")
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

# --- 전역 경로 설정 ---
PATH_JPGS = "/home/mmc-server3/Server/Datasets/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server3/Server/Datasets/rs19_val/uint8/rs19_val"
PATH_MODELS = "RailNet_DT/models"
NUM_CLASSES = 19

def create_model(output_channels=NUM_CLASSES):
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        num_labels=output_channels,
        ignore_mismatched_sizes=True
    )

def get_metrics_from_confusion_matrix(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / (union.astype(np.float32) + 1e-6)
    miou = np.nanmean(iou)
    return iou, miou

# ✅ [기능 복원] 텐서를 wandb 이미지로 변환하기 위한 전처리 해제 함수
def unnormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy() * 255

def main():
    ddp_setup()
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if rank == 0:
        wandb.init(project="DP_train_full", config=args)
        if not os.path.exists(PATH_MODELS): os.makedirs(PATH_MODELS)
    
    per_gpu_batch_size = args.batch_size // world_size
    assert args.batch_size % world_size == 0, "배치 사이즈는 GPU 개수로 나누어 떨어져야 합니다."

    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    if args.optimizer == 'adam': optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adagrad': optimizer = Adagrad(model.parameters(), lr=args.learning_rate)
    
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    image_processor = SegformerImageProcessor(size={"height": 1024, "width": 1024})
    
    train_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Train', NUM_CLASSES)
    valid_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Valid', NUM_CLASSES)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=per_gpu_batch_size, shuffle=False, drop_last=True, sampler=valid_sampler, num_workers=4)
    
    best_miou = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        loader_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0))
        for inputs, masks in loader_tqdm:
            inputs, masks = inputs.to(rank), masks.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            logits = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        local_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        val_log_data = {} # 시각화를 위해 첫 검증 배치를 저장할 변수
        with torch.no_grad():
            for i, (inputs, masks) in enumerate(valid_loader):
                inputs, masks = inputs.to(rank), masks.to(rank)
                outputs = model(inputs).logits
                logits = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                
                preds = torch.argmax(logits, dim=1)
                
                mask = (masks != 255)
                hist = np.bincount(NUM_CLASSES * masks[mask].cpu().numpy().astype(int) + preds[mask].cpu().numpy(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
                local_confusion_matrix += hist

                if i == 0: # 첫 번째 배치만 저장
                    val_log_data['images'] = inputs
                    val_log_data['gts'] = masks.cpu().numpy()
                    val_log_data['preds'] = preds.cpu().numpy()

        local_confusion_matrix_tensor = torch.from_numpy(local_confusion_matrix).to(rank)
        dist.all_reduce(local_confusion_matrix_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            global_confusion_matrix = local_confusion_matrix_tensor.cpu().numpy()
            class_ious, final_miou = get_metrics_from_confusion_matrix(global_confusion_matrix)
            
            # ✅ [수정] 변수 이름을 'loader'에서 'train_loader'로 수정
            avg_loss = total_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train loss: {avg_loss:.4f} | Val MIoU: {final_miou:.4f} | lr: {current_lr:.6f}")
            
            log_payload = {
                "train_loss": avg_loss,
                "val_MIoU": final_miou,
                "lr": current_lr, 
                "epoch": epoch
            }
            
            # ✅ [기능 복원] 클래스별 IoU 로깅
            for i, iou in enumerate(class_ious):
                log_payload[f'val_iou/{CLASS_ID_TO_NAME[i]}'] = iou
            
            # ✅ [기능 복원] wandb에 샘플 이미지 로깅
            log_images = []
            for i in range(min(4, per_gpu_batch_size)):
                img = unnormalize_image(val_log_data['images'][i])
                gt_mask = val_log_data['gts'][i]
                pred_mask = val_log_data['preds'][i]
                
                log_images.append(wandb.Image(
                    img,
                    masks={
                        "ground_truth": {"mask_data": gt_mask, "class_labels": CLASS_ID_TO_NAME},
                        "prediction": {"mask_data": pred_mask, "class_labels": CLASS_ID_TO_NAME}
                    }
                ))
            log_payload["validation_samples"] = log_images
            
            wandb.log(log_payload)

            if final_miou > best_miou:
                best_miou = final_miou
                torch.save(model.module.state_dict(), os.path.join(PATH_MODELS, f'sweep_{wandb.run.id}_best.pth'))
                print(f"Epoch {epoch+1}: Best model saved with MIoU: {best_miou:.4f}")
        
        dist.barrier()
        scheduler.step()
    
    if rank == 0:
        wandb.finish()
        
    ddp_cleanup()

if __name__ == "__main__":
    main()