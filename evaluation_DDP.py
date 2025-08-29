import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 사용자 정의 모듈 및 transformers
from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerImageProcessor

# --- 1. 경로 및 설정 ---
PATH_JPGS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
PATH_MODEL_PTH = "/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"
NUM_CLASSES = 13  # The pretrained model has 13 classes
BATCH_SIZE = 8

# --- 2. 클래스 정보 및 리매핑 설정 ---
CLASSES_TO_TRAIN = [0, 1, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16]
REMAPPED_ID_TO_NAME = {
    0: "background", 1: "rail-track", 2: "rail-bed", 3: "rail-platform", 4: "rail-pole",
    5: "vehicle-car", 6: "vehicle-truck", 7: "vehicle-train", 8: "person", 9: "vegetation",
    10: "terrain", 11: "sky", 12: "building"
}
# Use a NumPy array for safe remapping on the CPU
remapping_array = np.full((256,), 255, dtype=np.uint8)
for new_id, old_id in enumerate(CLASSES_TO_TRAIN):
    remapping_array[old_id] = new_id

# --- DDP 설정 함수 ---
def ddp_setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

def ddp_cleanup():
    dist.destroy_process_group()

# --- 3. 평가를 위한 모델 로드 함수 ---
def load_model_for_evaluation(pth_path):
    print(f"Step 1: Loading model object directly from {pth_path}...")
    try:
        model = torch.load(pth_path, map_location="cpu", weights_only=False)
        if isinstance(model, nn.DataParallel):
            model = model.module
        print("✅ Model object successfully loaded!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model object: {e}")
        raise

# --- 4. 클래스별 IoU 및 MIoU 계산 함수 ---
def get_metrics(confusion_matrix, class_names):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    miou = np.nanmean(iou)
    per_class_iou = {name: iou_val for name, iou_val in zip(class_names.values(), iou)}
    return miou, per_class_iou

# --- 메인 실행 함수 ---
def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ddp_setup()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    model = load_model_for_evaluation(PATH_MODEL_PTH).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    image_processor = SegformerImageProcessor(size={"height": 1024, "width": 1024})
    valid_dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, [1024, 1024], 'Valid', val_fraction=0.2, num_labels=19)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE // world_size, shuffle=False, sampler=valid_sampler, num_workers=4)

    model.eval()
    local_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loader_tqdm = tqdm(valid_loader, desc="Evaluating", disable=(rank != 0))
    
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(loader_tqdm):
            inputs = inputs.to(local_rank)
            
            # Perform safe remapping on CPU using NumPy
            masks_np = masks.numpy()
            gts_np = remapping_array[masks_np]
            gts = torch.from_numpy(gts_np).to(local_rank)

            outputs = model(pixel_values=inputs)
            logits = getattr(outputs, 'logits', outputs)

            logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            mask = (gts_np != 255)
            hist = np.bincount(NUM_CLASSES * gts_np[mask].astype(int) + preds[mask], minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            local_confusion_matrix += hist

    dist.all_reduce(torch.from_numpy(local_confusion_matrix).to(local_rank), op=dist.ReduceOp.SUM)

    if rank == 0:
        print("\n--- Evaluation Finished ---")
        global_confusion_matrix = local_confusion_matrix
        final_miou, per_class_iou = get_metrics(global_confusion_matrix, REMAPPED_ID_TO_NAME)

        print(f"📈 Overall MIoU: {final_miou:.4f}")
        for class_name, iou in per_class_iou.items():
            print(f"  - Class '{class_name}': IoU = {iou:.4f}")
        
        print("✅ Results logged to console.")

    ddp_cleanup()

if __name__ == "__main__":
    main()