import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import torch.nn as nn # nn.DataParallel을 확인하기 위해 import

# 기존 학습 코드에서 사용한 데이터셋 클래스와 이미지 프로세서를 가져옵니다.
from scripts.dataloader_SegFormer import CustomDataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- 경로 및 설정 (사용자 환경에 맞게 수정) ---
PATH_TEST_JPGS = "/home/mmc-server3/Server/Datasets/rs19_val/jpgs/rs19_val"
PATH_TEST_MASKS = "/home/mmc-server3/Server/Datasets/rs19_val/uint8/rs19_val"
PATH_MODEL_PTH = "/home/mmc-server3/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"
PATH_OUTPUTS = "/home/mmc-server3/RailSafeNet_mini_DT/output"
NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAILSEM_COLOR_MAP = [
    [0, 0, 0], [111, 74, 0], [70, 70, 70], [128, 64, 128], [244, 35, 232],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [107, 142, 35], [152, 251, 152],
    [220, 220, 0], [250, 170, 30], [102, 102, 156], [190, 153, 153]
]

def main():
    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    # ✅ 1. .pth 파일과 동일한 'SegFormer-B3' 아키텍처를 생성합니다.
    print("Step 1: Creating a clean SegFormer-B3 model with a new 19-class head...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b3-finetuned-cityscapes-1024-1024", # ✅ B0가 아닌 B3로 수정
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    # 2. .pth 파일에서 state_dict를 추출합니다.
    print(f"Step 2: Loading weights from {PATH_MODEL_PTH}...")
    loaded_object = torch.load(PATH_MODEL_PTH, map_location=DEVICE, weights_only=False)
    
    # DataParallel 래퍼를 확인하고 실제 state_dict를 가져옵니다.
    if isinstance(loaded_object, nn.DataParallel):
        state_dict = loaded_object.module.state_dict()
    else:
        state_dict = loaded_object.state_dict()

    # 3. strict=False를 사용하여 B3 백본 가중치만 불러옵니다.
    #    - B3 백본 가중치는 이제 이름과 크기가 모두 일치하여 성공적으로 로드됩니다.
    #    - classifier 가중치는 클래스 수가 달라 무시됩니다.
    print("Step 3: Loading weights with strict=False...")
    model.load_state_dict(state_dict, strict=False)
    
    print("\nModel setup complete. The B3 backbone is pre-trained, but the classifier head is new.")
    model.to(DEVICE)
    model.eval()

    # --- 이하 추론 로직은 동일 ---
    image_processor = SegformerImageProcessor(size={"height": 1024, "width": 1024})
    test_dataset = CustomDataset(PATH_TEST_JPGS, PATH_TEST_MASKS, image_processor, [1024, 1024], 'Valid', NUM_CLASSES)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(DEVICE)
            outputs = model(pixel_values=inputs).logits
            
            upsampled_logits = F.interpolate(
                outputs,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1).cpu().numpy()

            for j in range(preds.shape[0]):
                original_filename = os.path.basename(test_dataset.image_paths[i * test_loader.batch_size + j])
                pred_mask = preds[j]
                color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                for class_id, color in enumerate(RAILSEM_COLOR_MAP):
                    color_mask[pred_mask == class_id] = color
                img_to_save = Image.fromarray(color_mask)
                save_path = os.path.join(PATH_OUTPUTS, original_filename.replace('.jpg', '.png'))
                img_to_save.save(save_path)

    print(f"Inference complete. Predictions are saved in '{PATH_OUTPUTS}'.")


if __name__ == "__main__":
    main()