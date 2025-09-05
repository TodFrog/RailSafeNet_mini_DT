# build_engine.py
import tensorrt as trt
import argparse

def build_engine(onnx_file_path, engine_file_path, use_fp16=False):
    """
    ONNX 모델 파일을 읽어 TensorRT 엔진을 빌드하고 저장합니다.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 1. TensorRT 빌더, 네트워크, 파서 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 2. ONNX 모델 파일 로드 및 파싱
    print(f"Loading ONNX file from: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file.")

    # 3. 빌드 환경설정
    # 최대 Workspace 메모리 설정 (GPU VRAM에 따라 조절)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

    # FP16 모드 활성화
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled.")
    else:
        print("FP16 not supported or not enabled.")

    # 4. 엔진 빌드
    print("Building TensorRT engine... (This may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
        return None
    print("Engine built successfully.")

    # 5. 엔진 파일로 저장
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to: {engine_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX file.")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model.")
    parser.add_argument("--engine", required=True, help="Path to save the TensorRT engine.")
    parser.add_argument("--fp16", action='store_true', help="Enable FP16 mode.")
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.fp16)