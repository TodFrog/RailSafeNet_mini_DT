import torch
from collections import OrderedDict

# ⚠️ 분석하려는 .pth 파일의 전체 경로를 여기에 입력하세요.
PATH_MODEL_PTH = "/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"

try:
    # 1. 모델 파일 불러오기
    # weights_only=False는 이전 코드와 동일하게 유지하여 원본 구조를 그대로 불러옵니다.
    loaded_object = torch.load(PATH_MODEL_PTH, map_location="cpu", weights_only=False)
    print("✅ 모델 파일을 성공적으로 불러왔습니다.")

    # DataParallel 등으로 래핑된 경우 실제 모델을 추출
    if hasattr(loaded_object, 'module'):
        state_dict = loaded_object.module.state_dict()
        print("✅ DataParallel 래핑을 확인하고 내부 모델의 state_dict를 추출했습니다.")
    else:
        # 모델 객체 자체일 경우 state_dict를 직접 추출
        state_dict = loaded_object.state_dict()
        print("✅ 모델 객체에서 직접 state_dict를 추출했습니다.")

    # 2. 최종 분류 레이어의 가중치 키(key) 찾기
    # SegFormer의 경우 보통 'decode_head.classifier.weight' 입니다.
    classifier_weight_key = 'decode_head.classifier.weight'
    
    if classifier_weight_key in state_dict:
        # 3. 가중치 텐서의 모양(shape) 확인
        weights = state_dict[classifier_weight_key]
        num_classes = weights.shape[0]
        
        print("\n--- 분석 결과 ---")
        print(f"'{classifier_weight_key}'의 가중치 텐서 모양: {weights.shape}")
        print(f"👉 이 모델이 학습된 원본 클래스의 개수는 **{num_classes}개** 입니다.")
        
    else:
        print("\n--- 분석 실패 ---")
        print(f"'{classifier_weight_key}' 키를 찾을 수 없습니다.")
        print("모델의 state_dict에 있는 키들을 직접 확인해보세요:")
        # 어떤 키가 있는지 상위 10개만 출력
        for i, key in enumerate(state_dict.keys()):
            if i >= 10: break
            print(f"- {key}")

except Exception as e:
    print(f"\n❌ 오류가 발생했습니다: {e}")
    print("파일 경로가 정확한지, 파일이 손상되지 않았는지 확인해주세요.")