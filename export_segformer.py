# export_segformer.py
import torch
from scripts.test_filtered_cls import load_model
import argparse

def main(args):
    # 1. 원본 PyTorch 모델 로드 (GPU 사용)
    print(f"Loading PyTorch model from: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path).to(device).eval()
    print("Model loaded successfully.")

    # 2. 모델에 들어갈 더미 입력 데이터 생성
    # 실제 추론 시 사용할 해상도와 동일하게 설정 (1, 3, 높이, 너비)
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    print(f"Creating dummy input with size: {dummy_input.shape}")

    # 3. ONNX 파일로 내보내기
    print(f"Exporting to ONNX format at: {args.onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11, # 호환성을 위한 버전 명시
            dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
        )
        print(f"SegFormer model successfully exported to {args.onnx_path}")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export SegFormer PyTorch model to ONNX.")
    parser.add_argument("--model_path", type=str, 
                        default='/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth',
                        help="Path to the input PyTorch model (.pth).")
    parser.add_argument("--onnx_path", type=str, default="segformer.onnx",
                        help="Path to save the output ONNX file.")
    parser.add_argument("--width", type=int, default=480,
                        help="Width of the input image for the ONNX model.")
    parser.add_argument("--height", type=int, default=270,
                        help="Height of the input image for the ONNX model.")
    args = parser.parse_args()
    
    main(args)