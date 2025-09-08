# export_segformer_21cls.py
import torch
import argparse
import os
import sys

# scripts 폴더를 path에 추가 (필요한 경우)
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from scripts.test_filtered_cls import load_model
except ImportError:
    print("Warning: Could not import from scripts.test_filtered_cls, trying direct torch.load")
    def load_model(path):
        return torch.load(path, map_location='cpu', weights_only=False)

def main(args):
    print(f"Loading PyTorch model from: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 21개 클래스 모델 로드
    try:
        model = load_model(args.model_path).to(device).eval()
    except Exception as e:
        print(f"Error loading model with load_model function: {e}")
        print("Trying direct torch.load...")
        model = torch.load(args.model_path, map_location=device, weights_only=False).eval()
    
    print("Model loaded successfully.")
    print(f"Model device: {next(model.parameters()).device}")

    # 더미 입력 생성 (21개 클래스 모델에 맞춤)
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    print(f"Creating dummy input with size: {dummy_input.shape}")

    # 모델 테스트 실행
    print("Testing model inference...")
    with torch.no_grad():
        try:
            test_output = model(dummy_input)
            if hasattr(test_output, 'logits'):
                print(f"Model output logits shape: {test_output.logits.shape}")
                num_classes = test_output.logits.shape[1]
                print(f"Detected number of classes: {num_classes}")
            else:
                print(f"Model output shape: {test_output.shape}")
        except Exception as e:
            print(f"Error during test inference: {e}")
            return

    # ONNX 내보내기
    print(f"Exporting to ONNX format at: {args.onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            do_constant_folding=True,  # 최적화 활성화
            verbose=False,
            # 동적 배치 크기 설정 (TensorRT 최적화에 유리)
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ SegFormer 21-class model successfully exported to {args.onnx_path}")
        
        # ONNX 파일 검증
        import onnx
        onnx_model = onnx.load(args.onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed")
        
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export SegFormer 21-class PyTorch model to ONNX.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the input PyTorch model (.pth) with 21 classes.")
    parser.add_argument("--onnx_path", type=str, default="segformer_21cls.onnx",
                        help="Path to save the output ONNX file.")
    parser.add_argument("--width", type=int, default=480,
                        help="Width of the input image for the ONNX model.")
    parser.add_argument("--height", type=int, default=270,
                        help="Height of the input image for the ONNX model.")
    args = parser.parse_args()
    
    main(args)