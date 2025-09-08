# build_engine_optimized.py
import tensorrt as trt
import argparse
import os

def build_engine(onnx_file_path, engine_file_path, use_fp16=False, workspace_size=2, batch_size=1):
    """
    ONNX 모델 파일을 읽어 최적화된 TensorRT 엔진을 빌드하고 저장합니다.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"🚀 Building TensorRT engine from: {onnx_file_path}")
    print(f"📝 Configuration:")
    print(f"   - FP16: {use_fp16}")
    print(f"   - Workspace size: {workspace_size}GB")
    print(f"   - Batch size: {batch_size}")

    # 1. TensorRT 빌더, 네트워크, 파서 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 2. ONNX 모델 파일 로드 및 파싱
    if not os.path.exists(onnx_file_path):
        print(f"❌ Error: ONNX file not found at {onnx_file_path}")
        return None
        
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(f"   {parser.get_error(error)}")
            return None
    print("✅ Completed parsing ONNX file.")

    # 3. 빌드 환경설정
    # Workspace 메모리 설정 (더 큰 메모리로 최적화 향상)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size << 30)
    
    # FP16 모드 활성화 (속도 대폭 향상)
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 mode enabled - expect 2x speed improvement")
    else:
        print("⚠️  FP16 not supported or not enabled")

    # 최적화 프로파일 설정 (세그멘테이션 모델에 최적화)
    profile = builder.create_optimization_profile()
    
    # 입력 shape 최적화 (배치 크기와 해상도 고정)
    input_tensor = network.get_input(0)
    input_shape = (batch_size, 3, 270, 480)  # 고정 해상도로 최적화
    
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    
    # 추가 최적화 플래그
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # 타입 안정성
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)  # 정밀도 최적화
    
    # 레이어 최적화 (세그멘테이션 모델에 유리)
    if hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    
    print("🔧 Optimization settings applied")

    # 4. 엔진 빌드
    print("⏳ Building TensorRT engine... (This may take 5-15 minutes)")
    print("   Please be patient, this process optimizes the model for your GPU")
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ Failed to build the engine.")
        print("💡 Try reducing workspace size or disabling FP16")
        return None
    
    print("✅ Engine built successfully!")

    # 5. 엔진 파일로 저장
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    file_size_mb = os.path.getsize(engine_file_path) / (1024 * 1024)
    print(f"💾 Engine saved to: {engine_file_path}")
    print(f"📊 Engine file size: {file_size_mb:.1f} MB")
    
    return engine_file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build optimized TensorRT engine from ONNX file.")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model.")
    parser.add_argument("--engine", required=True, help="Path to save the TensorRT engine.")
    parser.add_argument("--fp16", action='store_true', help="Enable FP16 mode for 2x speed boost.")
    parser.add_argument("--workspace", type=int, default=2, help="Workspace size in GB (default: 2)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for optimization (default: 1)")
    args = parser.parse_args()

    result = build_engine(args.onnx, args.engine, args.fp16, args.workspace, args.batch)
    
    if result:
        print("🎉 TensorRT engine build completed successfully!")
        print(f"🚀 Ready to use: {result}")
    else:
        print("💥 Engine build failed!")
        exit(1)