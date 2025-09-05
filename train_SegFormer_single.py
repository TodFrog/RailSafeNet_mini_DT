#!/usr/bin/env python3
# train_SegFormer_single.py - 시각화 기능 추가 버전

import argparse
from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import wandb
from tqdm import tqdm
import time
import gc
import matplotlib.pyplot as plt
import cv2

# 클래스 정보와 IoU 계산 함수
NUM_CLASSES = 22
IGNORE_INDEX = 21
CLASS_ID_TO_NAME = {
    0: 'buffer-stop', 1: 'crossing', 2: 'guard-rail', 3: 'train-car', 4: 'platform',
    5: 'rail', 6: 'switch-indicator', 7: 'switch-left', 8: 'switch-right', 9: 'switch-unknown',
    10: 'switch-static', 11: 'track-sign-front', 12: 'track-signal-front', 13: 'track-signal-back',
    14: 'person-group', 15: 'car', 16: 'fence', 17: 'person', 18: 'pole', 19: 'rail-occluder', 20: 'truck'
}
RAIL_CLASS_ID = 5

# RailSem19 클래스별 시각화 색상
RAILSEM19_COLORS = {
    0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156], 4: [190, 153, 153],
    5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35], 9: [152, 251, 152],
    10: [70, 130, 180], 11: [220, 20, 60], 12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70],
    15: [0, 60, 100], 16: [0, 80, 100], 17: [0, 0, 230], 18: [119, 11, 32], 19: [0, 0, 0], 20: [0, 0, 142]
}

def get_ious(cm):
    """혼동 행렬로부터 mIoU와 클래스별 IoU를 계산"""
    cm_valid = cm[:IGNORE_INDEX, :IGNORE_INDEX]
    intersection = np.diag(cm_valid)
    union = cm_valid.sum(axis=1) + cm_valid.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    miou = np.nanmean(iou)
    return miou, iou

def create_visualization(image_tensor, gt_mask, pred_mask, alpha=0.7):
    """시각화 이미지 생성"""
    try:
        # 이미지 정규화 해제
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # 마스크를 CPU로 이동
        gt_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
        pred_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask
        
        h, w = gt_np.shape
        
        # GT 마스크 색상화
        gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in RAILSEM19_COLORS.items():
            if class_id < IGNORE_INDEX:
                gt_colored[gt_np == class_id] = color
        
        # 예측 마스크 색상화
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in RAILSEM19_COLORS.items():
            if class_id < IGNORE_INDEX:
                pred_colored[pred_np == class_id] = color
        
        # 오버레이 생성
        gt_overlay = cv2.addWeighted(image_np, 1-alpha, gt_colored, alpha, 0)
        pred_overlay = cv2.addWeighted(image_np, 1-alpha, pred_colored, alpha, 0)
        
        return image_np, gt_overlay, pred_overlay
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")
        return None, None, None

# 기본 설정
LIGHT = False
WANDB = True

if not LIGHT:
    PATH_JPGS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
    PATH_MASKS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
else:
    PATH_JPGS = "RailNet_DT/rs19_val_light/jpgs/rs19_val"
    PATH_MASKS = "RailNet_DT/rs19_val_light/uint8/rs19_val"

PATH_MODELS = "RailNet_DT/models"
PATH_LOGS = "RailNet_DT/logs"

os.makedirs(PATH_MODELS, exist_ok=True)
os.makedirs(PATH_LOGS, exist_ok=True)

def create_model(output_channels=NUM_CLASSES):
    """B3 모델 생성"""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b3-finetuned-ade-512-512",
        num_labels=output_channels,
        ignore_mismatched_sizes=True
    )
    
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

def freeze_encoder_partially(model, freeze_blocks=0):
    """인코더 부분 동결"""
    if freeze_blocks > 0:
        total_blocks = len(model.segformer.encoder.block)
        print(f"Freezing first {freeze_blocks}/{total_blocks} encoder blocks")
        
        for i, block in enumerate(model.segformer.encoder.block):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False
    else:
        print("All encoder blocks are trainable")

def train_with_accumulation(model, args):
    """메모리 최적화된 훈련 (시각화 포함)"""
    start = time.time()
    best_rail_iou = 0.0
    device = torch.device("cuda:0")
    
    image_size = [args.image_resolution, args.image_resolution]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    print(f"Training configuration:")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Accumulation steps: {args.accumulation_steps}")
    print(f"- Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"- Image resolution: {args.image_resolution}x{args.image_resolution}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Frozen encoder blocks: {args.freeze_encoder_blocks}")

    for epoch in range(args.epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        
        train_loss = 0
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        
        # 시각화 샘플 저장용
        vis_samples = None
        
        for phase in ['Train', 'Valid']:
            image_processor = SegformerImageProcessor(do_reduce_labels=False)
            dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, image_size, subset=phase, val_fraction=0.1)
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=(phase=='Train'), 
                drop_last=True, 
                num_workers=0,
                pin_memory=False
            )
            
            if phase == 'Train':
                model.train()
                optimizer.zero_grad()
                epoch_start = time.time()
                
                for batch_idx, (inputs, masks) in enumerate(tqdm(dataloader, desc="Training")):
                    try:
                        inputs = inputs.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True, dtype=torch.long)
                        masks = torch.clamp(masks, 0, IGNORE_INDEX)
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(pixel_values=inputs, labels=masks)
                            loss = outputs.loss / args.accumulation_steps
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        scaler.scale(loss).backward()
                        train_loss += loss.item() * args.accumulation_steps
                        
                        if (batch_idx + 1) % args.accumulation_steps == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                                
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM at batch {batch_idx}, clearing cache...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise e
                            
                epoch_time = time.time() - epoch_start
                print(f"Training epoch completed in {epoch_time:.1f}s")
                    
            elif phase == 'Valid':
                model.eval()
                
                with torch.no_grad():
                    for batch_idx, (inputs, masks) in enumerate(tqdm(dataloader, desc="Validation")):
                        try:
                            inputs = inputs.to(device, non_blocking=True)
                            masks = masks.to(device, non_blocking=True, dtype=torch.long)
                            masks = torch.clamp(masks, 0, IGNORE_INDEX)

                            with torch.cuda.amp.autocast():
                                outputs = model(pixel_values=inputs)
                                logits = nn.functional.interpolate(
                                    outputs.logits, size=masks.shape[-2:], 
                                    mode="bilinear", align_corners=False
                                )
                            
                            predictions = torch.argmax(logits, dim=1)
                            
                            # 혼동 행렬 업데이트
                            gt_masks_np = masks.cpu().numpy().flatten()
                            pred_masks_np = predictions.cpu().numpy().flatten()
                            
                            mask = (gt_masks_np >= 0) & (gt_masks_np < NUM_CLASSES)
                            
                            cm_update = np.bincount(
                                NUM_CLASSES * gt_masks_np[mask].astype(int) + pred_masks_np[mask],
                                minlength=NUM_CLASSES**2
                            ).reshape(NUM_CLASSES, NUM_CLASSES)
                            confusion_matrix += cm_update
                            
                            # 시각화 샘플 수집 (첫 번째 배치에서만, 매 5 epoch마다)
                            if batch_idx == 0 and (epoch + 1) % 5 == 0 and vis_samples is None:
                                sample_img = inputs[0].cpu()
                                sample_gt = masks[0].cpu()
                                sample_pred = predictions[0].cpu()
                                
                                orig_img, gt_overlay, pred_overlay = create_visualization(
                                    sample_img, sample_gt, sample_pred
                                )
                                
                                if orig_img is not None:
                                    vis_samples = {
                                        'original': orig_img,
                                        'ground_truth': gt_overlay,
                                        'prediction': pred_overlay
                                    }
                                    
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
        
        # 에포크 결과 계산
        avg_train_loss = train_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        mIoU, class_ious = get_ious(confusion_matrix)
        current_rail_iou = class_ious[RAIL_CLASS_ID]
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | mIoU: {mIoU:.4f} | Rail IoU: {current_rail_iou:.4f} | LR: {current_lr:.6f}')
        print(f'GPU Memory: {memory_allocated:.1f}GB')

        # 베스트 모델 저장
        if current_rail_iou > best_rail_iou:
            best_rail_iou = current_rail_iou
            best_model_name = f'best_rail_iou_{best_rail_iou:.4f}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), os.path.join(PATH_MODELS, best_model_name))
            print(f'🚀 Best model saved: {best_model_name}')

        # WandB 로깅
        if WANDB:
            log_payload = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val/mIoU": mIoU,
                "val_iou/rail": current_rail_iou,
                "val/best_rail_iou": best_rail_iou,  # 이게 sweep에서 추적할 메트릭
                "learning_rate": current_lr,
                "gpu_memory_gb": memory_allocated,
                "effective_batch_size": args.batch_size * args.accumulation_steps
            }
            
            # 클래스별 IoU 로깅
            for i, iou in enumerate(class_ious):
                class_name = CLASS_ID_TO_NAME.get(i, f"class_{i}")
                log_payload[f'val_iou/{class_name}'] = iou
            
            # 시각화 추가 (매 5 epoch마다)
            if vis_samples is not None:
                log_payload.update({
                    "segmentation/original": wandb.Image(vis_samples['original'], caption=f"Original - Epoch {epoch+1}"),
                    "segmentation/ground_truth": wandb.Image(vis_samples['ground_truth'], caption=f"Ground Truth - Epoch {epoch+1}"),
                    "segmentation/prediction": wandb.Image(vis_samples['prediction'], caption=f"Prediction - Epoch {epoch+1}")
                })
                print(f"Visualization logged for epoch {epoch+1}")
                
            wandb.log(log_payload)

        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start
    print(f'Training completed in {total_time/60:.1f} minutes')
    print(f'Best Rail IoU: {best_rail_iou:.4f}')
    
    return model

def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="SegFormer B3 Training with Sweep Support")
    
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--image_resolution', type=int, default=768)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--freeze_encoder_blocks', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_step', type=int, default=15)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.85)
    parser.add_argument('--epochs', type=int, default=40)
    
    args = parser.parse_args()
    
    if WANDB:
        wandb.init(project="SegFormer-B3-RailSem19-Sweep", config=vars(args))
        wandb.define_metric("val/mIoU", summary="max")
        wandb.define_metric("val_iou/rail", summary="max")
        wandb.define_metric("val/best_rail_iou", summary="max")  # 핵심 메트릭
    
    print(f"Starting optimized B3 training with {torch.cuda.device_count()} GPU(s)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Available GPU memory: {gpu_memory:.1f}GB")
    
    try:
        model = create_model()
        freeze_encoder_partially(model, args.freeze_encoder_blocks)
        
        model_final = train_with_accumulation(model, args)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()