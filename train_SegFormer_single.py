#!/usr/bin/env python3
# train_SegFormer_single_fixed.py - DDP 버전과 동일한 전처리 및 메트릭 적용

import argparse
from scripts.dataloader_SegFormer import CustomDataset
from scripts.metrics_filtered_cls import compute_map_cls, compute_IoU
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

# ===== DDP 버전과 동일한 설정 =====
# 필터링된 클래스 설정 (DDP 버전과 동일)
IGNORE_LIST = [0,1,2,6,8,9,15,16,19,20]  # 제외할 클래스
IGNORE_SET = set(IGNORE_LIST)
CLS_REMAINING = [num for num in range(0, 22) if num not in IGNORE_SET]  # [3,4,5,7,10,11,12,13,14,17,18,21]
NUM_CLASSES = len(CLS_REMAINING) + 1  # 12개 클래스 + 1개 배경 = 13개
BACKGROUND_ID = len(CLS_REMAINING)  # 12

# 리매핑된 클래스 정보
REMAPPED_CLASS_NAMES = {
    0: 'train-car',      # 원본 3
    1: 'platform',       # 원본 4  
    2: 'rail',           # 원본 5 ⭐ 이것이 rail-track 클래스
    3: 'switch-left',    # 원본 7
    4: 'switch-static',  # 원본 10
    5: 'track-sign-front', # 원본 11
    6: 'track-signal-front', # 원본 12
    7: 'track-signal-back',  # 원본 13
    8: 'person-group',   # 원본 14
    9: 'person',         # 원본 17
    10: 'pole',          # 원본 18
    11: 'truck',         # 원본 21
    12: 'background'     # 배경
}

RAIL_TRACK_ID = 2  # 리매핑 후 rail 클래스 ID

# RailSem19 원본 색상 (리매핑된 클래스용)
RAILSEM19_COLORS = {
    0: [102, 102, 156],  # train-car
    1: [190, 153, 153],  # platform
    2: [153, 153, 153],  # rail ⭐
    3: [220, 220, 0],    # switch-left
    4: [70, 130, 180],   # switch-static
    5: [220, 20, 60],    # track-sign-front
    6: [255, 0, 0],      # track-signal-front
    7: [0, 0, 142],      # track-signal-back
    8: [0, 0, 70],       # person-group
    9: [0, 0, 230],      # person
    10: [119, 11, 32],   # pole
    11: [0, 0, 142],     # truck
    12: [0, 0, 0]        # background
}

def create_visualization(image_tensor, gt_mask, pred_mask, alpha=0.7):
    """시각화 이미지 생성 (리매핑된 클래스용)"""
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
            if class_id < NUM_CLASSES:
                gt_colored[gt_np == class_id] = color
        
        # 예측 마스크 색상화
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in RAILSEM19_COLORS.items():
            if class_id < NUM_CLASSES:
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
    
    model.config.semantic_loss_ignore_index = BACKGROUND_ID
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
    """DDP 버전과 동일한 메트릭을 사용한 훈련"""
    start = time.time()
    best_rail_iou = 0.0
    device = torch.device("cuda:0")
    
    image_size = [args.image_resolution, args.image_resolution]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    print(f"Training configuration:")
    print(f"- Filtered classes: {len(CLS_REMAINING)} + 1 background = {NUM_CLASSES} total")
    print(f"- Rail-track class ID: {RAIL_TRACK_ID}")
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
        val_mAP_list, val_MmAP_list, val_IoU_list, val_MIoU_list = [], [], [], []
        classes_MAP, classes_AP, classes_IoU, classes_MIoU = {}, {}, {}, {}
        
        # 시각화 샘플 저장용
        vis_samples = None
        
        for phase in ['Train', 'Valid']:
            # ⭐ DDP 버전과 동일한 데이터로더 설정
            image_processor = SegformerImageProcessor(size={"height": args.image_resolution, "width": args.image_resolution})
            dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, image_size, subset=phase, val_fraction=0.1)  # ✅ 0.1로 수정
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
                        masks = torch.clamp(masks, 0, BACKGROUND_ID)
                        
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
                            print(f"OOM at batch {batch_idx} (batch_size={inputs.shape[0]})")
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            # 배치를 반으로 나누어 재시도
                            try:
                                mid = inputs.shape[0] // 2
                                if mid > 0:
                                    print(f"Retrying with reduced batch size: {mid}")
                                    inputs_1, masks_1 = inputs[:mid], masks[:mid]
                                    inputs_2, masks_2 = inputs[mid:], masks[mid:]
                                    
                                    # 첫 번째 반
                                    with torch.cuda.amp.autocast():
                                        outputs_1 = model(pixel_values=inputs_1, labels=masks_1)
                                        loss_1 = outputs_1.loss / (args.accumulation_steps * 2)
                                    scaler.scale(loss_1).backward()
                                    
                                    # 두 번째 반
                                    with torch.cuda.amp.autocast():
                                        outputs_2 = model(pixel_values=inputs_2, labels=masks_2)
                                        loss_2 = outputs_2.loss / (args.accumulation_steps * 2)
                                    scaler.scale(loss_2).backward()
                                    
                                    train_loss += (loss_1.item() + loss_2.item()) * args.accumulation_steps
                                    print("✅ OOM recovered by batch splitting")
                                else:
                                    print("❌ Cannot split batch further, skipping...")
                                    continue
                            except RuntimeError:
                                print("❌ OOM persists even with batch splitting, skipping...")
                                torch.cuda.empty_cache()
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
                            masks = torch.clamp(masks, 0, BACKGROUND_ID)

                            with torch.cuda.amp.autocast():
                                outputs = model(pixel_values=inputs)
                                logits = nn.functional.interpolate(
                                    outputs.logits, size=masks.shape[-2:], 
                                    mode="bilinear", align_corners=False
                                )
                            
                            predictions = torch.argmax(logits, dim=1)
                            
                            # ⭐ DDP 버전과 동일한 메트릭 계산
                            gt_masks = masks.cpu().detach().numpy()
                            pred_masks = predictions.cpu().detach().numpy()
                            
                            for gt, prediction in zip(gt_masks, pred_masks):
                                # DDP 버전과 동일한 메트릭 함수 사용
                                mAP, classes_AP = compute_map_cls(gt, prediction, classes_AP)
                                Mmap, classes_MAP = compute_map_cls(gt, prediction, classes_MAP, major=True)
                                IoU, _, _, _, classes_IoU = compute_IoU(gt, prediction, classes_IoU)
                                MIoU, _, _, _, classes_MIoU = compute_IoU(gt, prediction, classes_MIoU, major=True)
                                
                                val_mAP_list.append(mAP)
                                val_MmAP_list.append(Mmap)
                                val_IoU_list.append(IoU)
                                val_MIoU_list.append(MIoU)
                            
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
        
        # ⭐ DDP 버전과 동일한 에포크 결과 계산
        avg_train_loss = train_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        # 에포크 평균 계산
        val_MmAP_avg = np.nanmean(val_MmAP_list) if val_MmAP_list else 0
        val_mAP_avg = np.nanmean(val_mAP_list) if val_mAP_list else 0
        val_MIoU_avg = np.nanmean(val_MIoU_list) if val_MIoU_list else 0
        val_IoU_avg = np.nanmean(val_IoU_list) if val_IoU_list else 0
        
        # 클래스별 평균 계산 (DDP와 동일)
        for cls, value in classes_MAP.items():
            classes_MAP[cls] = np.divide(value[0], value[1])
        classes_MmAP_all = np.mean(np.array(list(classes_MAP.values())), axis=0) if classes_MAP else 0
        
        for cls, value in classes_AP.items():
            classes_AP[cls] = np.divide(value[0], value[1])
        classes_mAP_all = np.mean(np.array(list(classes_AP.values())), axis=0) if classes_AP else 0
        
        for cls, value in classes_IoU.items():
            classes_IoU[cls] = np.divide(value[0], value[1])
        classes_IoU_all = np.mean(np.array(list(classes_IoU.values()))[:, :4], axis=0) if classes_IoU else [0,0,0,0]
        
        for cls, value in classes_MIoU.items():
            classes_MIoU[cls] = np.divide(value[0], value[1])
        classes_MIoU_all = np.mean(np.array(list(classes_MIoU.values()))[:, :4], axis=0) if classes_MIoU else [0,0,0,0]
        
        # Rail-track IoU 추출 (클래스 ID 2)
        current_rail_iou = classes_IoU.get(RAIL_TRACK_ID, [0])[0] if RAIL_TRACK_ID in classes_IoU else 0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        
        # DDP 버전과 동일한 출력 형식
        print(f'Epoch {epoch+1}: Train loss: {avg_train_loss:.4f} | lr: {current_lr:.6f} | mAP: {classes_mAP_all:.4f} | MmAP: {classes_MmAP_all:.4f} | IoU: {classes_IoU_all[0]:.4f} | MIoU: {classes_MIoU_all[0]:.4f}')
        print(f'Rail-track IoU: {current_rail_iou:.4f} | GPU Memory: {memory_allocated:.1f}GB')

        # 베스트 모델 저장
        if current_rail_iou > best_rail_iou:
            best_rail_iou = current_rail_iou
            best_model_name = f'best_rail_iou_{best_rail_iou:.4f}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), os.path.join(PATH_MODELS, best_model_name))
            print(f'🚀 Best model saved: {best_model_name}')

        # WandB 로깅 (DDP와 동일한 메트릭명 사용)
        if WANDB:
            log_payload = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "lr": current_lr,
                "mAP": classes_mAP_all,
                "MmAP": classes_MmAP_all,
                "IoU": classes_IoU_all[0],
                "MIoU": classes_MIoU_all[0],
                "val_iou/rail-track": current_rail_iou,  # ⭐ DDP와 동일한 메트릭명
                "best_rail_iou": best_rail_iou,
                "gpu_memory_gb": memory_allocated,
                "effective_batch_size": args.batch_size * args.accumulation_steps
            }
            
            # 개별 클래스 IoU 로깅
            for class_id, class_name in REMAPPED_CLASS_NAMES.items():
                if class_id in classes_IoU:
                    log_payload[f'val_iou/{class_name}'] = classes_IoU[class_id][0]
            
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
    print(f'Best Rail-track IoU: {best_rail_iou:.4f}')
    
    return model

def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="SegFormer B3 Training with DDP-Compatible Metrics")
    
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
        wandb.init(project="SegFormer-B3-RailSem19-DDP-Compatible", config=vars(args))
        wandb.define_metric("val_iou/rail-track", summary="max")
        wandb.define_metric("best_rail_iou", summary="max")
        wandb.define_metric("MIoU", summary="max")
    
    print(f"Starting DDP-compatible B3 training with filtered classes")
    print(f"Rail-track class will be tracked as ID {RAIL_TRACK_ID}")
    
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