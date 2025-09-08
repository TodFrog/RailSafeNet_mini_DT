# run_hybrid_21cls.py

import cv2
import os
import time
import numpy as np
import argparse
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque

# -------------------------------------------------------------------
# TensorRT 추론 클래스
# -------------------------------------------------------------------
class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 엔진 로드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 입출력 바인딩 설정
        self.input_binding = 0
        self.output_binding = 1
        
        # 메모리 할당
        self.input_shape = self.engine.get_binding_shape(self.input_binding)
        self.output_shape = self.engine.get_binding_shape(self.output_binding)
        
        self.input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
        
        # GPU 메모리 할당
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        # CUDA 스트림
        self.stream = cuda.Stream()
        
        print(f"TensorRT engine loaded successfully")
        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
    
    def infer(self, input_data):
        # 입력 데이터를 GPU로 복사
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        
        # 추론 실행
        self.context.execute_async_v2([int(self.d_input), int(self.d_output)], self.stream.handle)
        
        # 결과를 CPU로 복사
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output_data

# -------------------------------------------------------------------
# 21개 클래스용 매핑 정의
# -------------------------------------------------------------------
# RailSem19 21개 클래스에서 철도 관련 클래스 ID 매핑
CLASS_MAPPING_21 = {
    'rail_track': [3, 4],      # rail track 클래스들
    'rail_bed': [5, 6],        # rail bed/ballast 클래스들  
    'rail_metal': [9, 10],     # rail 메탈 부분
    'background': 20           # 배경 클래스
}

# -------------------------------------------------------------------
# 헬퍼 함수들 (21개 클래스용으로 수정)
# -------------------------------------------------------------------
def find_extreme_y_values(arr, values=None):
    if values is None:
        values = CLASS_MAPPING_21['rail_track'] + CLASS_MAPPING_21['rail_bed']
    mask = np.isin(arr, values)
    rows_with_values = np.any(mask, axis=1)
    y_indices = np.nonzero(rows_with_values)[0]
    if y_indices.size == 0:
        return None, None
    return y_indices[0], y_indices[-1]

def filter_crossings(image, edges_dict):
    filtered_edges = {}
    for key, values in edges_dict.items():
        if not values: continue
        merged = [values[0]]
        for start, end in values[1:]:
            if start - merged[-1][1] < 50:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        filtered_edges[key] = merged
    return filtered_edges

def find_edges(image, y_levels, values=None, min_width=19):
    if values is None:
        values = CLASS_MAPPING_21['rail_track'] + CLASS_MAPPING_21['rail_bed']
    
    edges_dict = {}
    for y in y_levels:
        if y >= image.shape[0]: continue
        row = image[y, :]
        mask = np.isin(row, values).astype(int)
        padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
        diff = np.diff(padded_mask)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        filtered_edges = [(start, end) for start, end in zip(starts, ends) 
                         if end - start + 1 >= min_width and 0 not in (start, end) 
                         and image.shape[1]-1 not in (start, end)]
        if filtered_edges:
            edges_dict[y] = filtered_edges
    return filter_crossings(image, edges_dict)

def find_rails(arr, y_levels, values=None, min_width=5):
    if values is None:
        values = CLASS_MAPPING_21['rail_metal']
    
    edges_all = []
    for y in y_levels:
        if y >= arr.shape[0]: continue
        row = arr[y, :]
        mask = np.isin(row, values).astype(int)
        padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
        diff = np.diff(padded_mask)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        filtered_edges = [(start, end) for start, end in zip(starts, ends) 
                         if end - start + 1 >= min_width and 0 not in (start, end) 
                         and arr.shape[1]-1 not in (start, end)]
        edges_all.extend(filtered_edges)
    return edges_all

def robust_rail_sides(border, threshold=7):
    border = np.array(border)
    if border.size < 4: return border
    border = border[border[:, 1] != border[:, 1].max()]  # 이미지 하단 제거
    if border.size < 4: return border
    
    steps_x = np.diff(border[:, 0])
    median_step = np.median(np.abs(steps_x))
    threshold_step = np.abs(threshold * np.abs(median_step))
    if threshold_step == 0: threshold_step = 20
    
    threshold_overcommings = abs(steps_x) > abs(threshold_step)
    if not np.any(threshold_overcommings): return border
    
    overcommings_indices = np.where(threshold_overcommings)[0]
    split_indices = np.concatenate(([0], overcommings_indices + 1, [len(border)]))
    best_segment = max([border[split_indices[i]:split_indices[i+1]] 
                       for i in range(len(split_indices)-1)], key=len)
    return best_segment

def find_rail_sides(img, edges_dict):
    left_border, right_border = [], []
    for y, xs in edges_dict.items():
        if not xs: continue
        rails = find_rails(img, [y], min_width=5)
        left_border_actual = [min(xs)[0], y]
        right_border_actual = [max(xs)[1], y]
        
        for zone in rails:
            if abs(zone[1] - left_border_actual[0]) < y * 0.04: 
                left_border_actual[0] = zone[0]
            if abs(zone[0] - right_border_actual[0]) < y * 0.04: 
                right_border_actual[0] = zone[1]
        
        left_border.append(left_border_actual)
        right_border.append(right_border_actual)
    
    left_border = robust_rail_sides(left_border)
    right_border = robust_rail_sides(right_border)
    return left_border, right_border

def bresenham_line(x0, y0, x1, y1):
    line = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        line.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return line

def interpolate_end_points(end_points_dict):
    line_arr = []
    points = sorted(end_points_dict.items())
    for i in range(len(points) - 1):
        y1, x1 = points[i]
        y2, x2 = points[i+1]
        line = np.array(bresenham_line(x1, y1, x2, y2))
        if np.any(line[:, 0] < 0): line = line[line[:, 0] > 0]
        line_arr.extend(list(line))
    return np.array(line_arr) if line_arr else np.array([])

def find_dist_from_edges(edges_dict, left_border, right_border, real_life_width_mm, real_life_target_mm):
    diffs_width = {k: max(e-s for s, e in v) for k, v in edges_dict.items() if v}
    scale_factors = {k: real_life_width_mm / v for k, v in diffs_width.items() if v > 0}
    target_distances_px = {k: int(real_life_target_mm / v) for k, v in scale_factors.items()}
    
    end_points_left, end_points_right = {}, {}
    for point in left_border:
        y = point[1]
        if y in target_distances_px:
            end_points_left[y] = point[0] - target_distances_px[y]
    for point in right_border:
        y = point[1]
        if y in target_distances_px:
            end_points_right[y] = point[0] + target_distances_px[y]
    return end_points_left, end_points_right

def find_zone_border(id_map, edges, irl_width_mm=1435, irl_target_mm=1000):
    left_border_pts, right_border_pts = find_rail_sides(id_map, edges)
    end_points_left, end_points_right = find_dist_from_edges(edges, left_border_pts, right_border_pts, irl_width_mm, irl_target_mm)
    border_l = interpolate_end_points(end_points_left)
    border_r = interpolate_end_points(end_points_right)
    return [border_l.tolist(), border_r.tolist()]

def get_clues(segmentation_mask, number_of_clues):
    lowest, highest = find_extreme_y_values(segmentation_mask)
    if lowest is not None and highest is not None and highest > lowest:
        clue_step = int((highest - lowest) / (number_of_clues + 1))
        if clue_step == 0: clue_step = 1
        return [highest - (i * clue_step) for i in range(number_of_clues)] + [lowest]
    return []

def border_handler(id_map, edges, target_distances):
    borders = []
    for target in target_distances:
        borders.append(find_zone_border(id_map, edges, irl_target_mm=target))
    return borders

def identify_ego_track(edges_dict, image_width):
    ego_edges_dict = {}
    last_ego_track_center = None
    sorted_y_levels = sorted(edges_dict.keys(), reverse=True)
    image_center_x = image_width / 2
    
    if sorted_y_levels:
        first_y = sorted_y_levels[0]
        tracks_at_first_y = edges_dict.get(first_y, [])
        if tracks_at_first_y:
            closest_track = min(tracks_at_first_y, key=lambda t: abs(((t[0] + t[1]) / 2) - image_center_x))
            ego_edges_dict[first_y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2

    for y in sorted_y_levels[1:]:
        if last_ego_track_center is None: continue
        tracks_at_y = edges_dict.get(y, [])
        if tracks_at_y:
            closest_track = min(tracks_at_y, key=lambda t: abs(((t[0] + t[1]) / 2) - last_ego_track_center))
            ego_edges_dict[y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2
            
    return ego_edges_dict

def manage_detections(results):
    accepted_moving = {0, 1, 2, 3, 7, 15, 16, 17, 18, 19}
    boxes_moving = []
    if results:
        for res in results:
            if res.boxes:
                for box in res.boxes:
                    if int(box.cls) in accepted_moving:
                        boxes_moving.append(box)
    return boxes_moving

def classify_detections(boxes_moving, borders, names):
    boxes_info = []
    colors_bgr = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]  # Yellow, Orange, Red
    safe_color = (0, 255, 0)  # Green

    for box in boxes_moving:
        x, y, w, h = box.xywh[0]
        criticality = -1
        color = safe_color
        
        bottom_center_point = (int(x), int(y + h / 2))

        for i, border_pair in enumerate(reversed(borders)):
            border_l = np.array(border_pair[0], dtype=np.int32)
            border_r = np.array(border_pair[1], dtype=np.int32)
            if border_l.size > 0 and border_r.size > 0:
                poly_points = np.concatenate((border_l, border_r[::-1]), axis=0)
                if cv2.pointPolygonTest(poly_points, bottom_center_point, False) >= 0:
                    criticality = len(borders) - 1 - i
                    color = colors_bgr[criticality]
                    break
        
        boxes_info.append({
            "xywh": box.xywh[0].cpu().numpy().tolist(), 
            "conf": box.conf[0].cpu().numpy().item(), 
            "cls_name": names[int(box.cls)], 
            "color": color
        })
    return boxes_info

# -------------------------------------------------------------------
# 전처리 및 후처리 함수
# -------------------------------------------------------------------
def preprocess_frame(frame, target_size=(270, 480)):
    """프레임을 TensorRT 입력 형식으로 전처리"""
    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 리사이즈
    resized = cv2.resize(frame_rgb, target_size[::-1])  # (w, h)
    
    # 정규화
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    # HWC to CHW
    transposed = normalized.transpose(2, 0, 1)
    
    # 배치 차원 추가
    batched = np.expand_dims(transposed, axis=0)
    
    return batched.astype(np.float32)

def postprocess_segmentation(output, original_size):
    """TensorRT 출력을 세그멘테이션 맵으로 후처리"""
    # 소프트맥스 적용
    softmax_output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
    
    # argmax로 클래스 예측
    id_map = np.argmax(softmax_output, axis=1).squeeze().astype(np.uint8)
    
    # 원본 크기로 리사이즈
    id_map_resized = cv2.resize(id_map, original_size, interpolation=cv2.INTER_NEAREST)
    
    return id_map_resized

# -------------------------------------------------------------------
# 메인 추론 및 시각화 함수
# -------------------------------------------------------------------
def run_inference_and_draw(frame, trt_seg, model_det, target_distances, num_ys=15):
    """21개 클래스 모델로 추론과 시각화를 수행합니다."""
    
    original_h, original_w = frame.shape[:2]
    
    # 1. SegFormer TensorRT 추론
    input_data = preprocess_frame(frame)
    seg_output = trt_seg.infer(input_data)
    id_map = postprocess_segmentation(seg_output, (original_w, original_h))
    
    # 2. 21개 클래스 기준 후처리
    clues = get_clues(id_map, num_ys)
    edges = find_edges(id_map, clues, min_width=int(original_w * 0.01))
    ego_edges = identify_ego_track(edges, id_map.shape[1])
    borders = border_handler(id_map, ego_edges, target_distances)
    
    # 3. YOLO 추론
    results = model_det(frame, verbose=False) 
    boxes_moving = manage_detections(results)
    classification = classify_detections(boxes_moving, borders, results[0].names)

    # 4. 시각화
    overlay = frame.copy()
    alpha = 0.2
    colors_bgr = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]

    # 안전 구역 시각화
    for i, border_pair in enumerate(reversed(borders)):
        border_l, border_r = np.array(border_pair[0], dtype=np.int32), np.array(border_pair[1], dtype=np.int32)
        if border_l.size > 0 and border_r.size > 0:
            poly_points = np.concatenate((border_l, border_r[::-1]), axis=0)
            cv2.fillPoly(overlay, [poly_points], colors_bgr[i])

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 경계선 그리기
    for i, border_pair in enumerate(reversed(borders)):
        border_l, border_r = np.array(border_pair[0], dtype=np.int32), np.array(border_pair[1], dtype=np.int32)
        if border_l.size > 0: cv2.polylines(frame, [border_l], isClosed=False, color=colors_bgr[i], thickness=2)
        if border_r.size > 0: cv2.polylines(frame, [border_r], isClosed=False, color=colors_bgr[i], thickness=2)

    # 객체 탐지 결과 그리기
    for box in classification:
        x_center, y_center, w, h = box["xywh"]
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
        label = f'{box["cls_name"]} {box["conf"]:.2f}'
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box["color"], 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), box["color"], -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    return frame

# -------------------------------------------------------------------
# 메인 실행 로직
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['ULTRALYTICS_LOGGING_LEVEL'] = 'ERROR'

    parser = argparse.ArgumentParser(description="RailSafeNet 21-class TensorRT Performance Demo.")
    parser.add_argument("--input", type=str, required=True, help="Input video file or camera index")
    parser.add_argument("--seg_engine", type=str, required=True, help="Path to SegFormer TensorRT engine")
    parser.add_argument("--det_engine", type=str, required=True, help="Path to YOLO TensorRT engine")
    args = parser.parse_args()

    print("🚀 RailSafeNet 21-Class TensorRT Demo")
    print("="*50)

    # TensorRT 엔진 로드
    print("📥 Loading TensorRT engines...")
    try:
        trt_seg = TensorRTInference(args.seg_engine)
        print("✅ SegFormer TensorRT engine loaded")
    except Exception as e:
        print(f"❌ Failed to load SegFormer engine: {e}")
        exit(1)

    try:
        from ultralytics import YOLO
        model_det = YOLO(args.det_engine)
        print("✅ YOLO TensorRT engine loaded")
    except Exception as e:
        print(f"❌ Failed to load YOLO engine: {e}")
        exit(1)

    # 비디오 캡처 설정
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
        print(f"📹 Using camera {args.input}")
    else:
        cap = cv2.VideoCapture(args.input)
        print(f"📁 Using video file: {args.input}")

    if not cap.isOpened():
        print("❌ Failed to open video source")
        exit(1)

    # 성능 모니터링 설정
    fps_deque = deque(maxlen=30)
    target_distances = [650, 1000, 2000]  # mm
    
    print("🔥 Starting inference loop...")
    print("Press 'q' to quit")
    print("="*50)
    
    frame_count = 0
    total_time = 0
    
    # Warm-up (첫 몇 프레임은 느릴 수 있음)
    print("🔧 Warming up...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            try:
                _ = run_inference_and_draw(frame, trt_seg, model_det, target_distances)
            except Exception as e:
                print(f"⚠️ Warm-up frame failed: {e}")
    print("✅ Warm-up complete")
    
    # 메인 루프
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("📽️ End of video reached")
            break

        try:
            processed_frame = run_inference_and_draw(frame, trt_seg, model_det, target_distances)
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
            continue
        
        # FPS 계산
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_deque.append(fps)
        avg_fps = sum(fps_deque) / len(fps_deque)
        
        frame_count += 1
        total_time += (end_time - start_time)
        
        # FPS 표시
        cv2.putText(processed_frame, f"FPS: {avg_fps:.1f} | Frame: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f"TensorRT Optimized | 21 Classes", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # 화면 출력
        cv2.imshow('RailSafeNet 21-Class TensorRT Demo', processed_frame)

        # 종료 조건
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 스페이스바로 일시정지
            cv2.waitKey(0)

    # 성능 요약
    print("\n" + "="*50)
    print("📊 Performance Summary:")
    print(f"   Total frames processed: {frame_count}")
    if frame_count > 0:
        avg_fps_total = frame_count / total_time
        print(f"   Average FPS: {avg_fps_total:.2f}")
        print(f"   Average frame time: {1000/avg_fps_total:.1f}ms")
    print("="*50)

    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("🎬 Demo completed successfully!")