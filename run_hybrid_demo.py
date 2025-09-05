# run_hybrid_demo.py

import cv2
import os
import time
import numpy as np
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO # 공식 ultralytics 라이브러리 사용
from scripts.test_filtered_cls import load_model
from collections import deque
from torch.cuda.amp import autocast

# -------------------------------------------------------------------
# (모든 헬퍼 함수들을 여기에 통합)
# -------------------------------------------------------------------
def find_extreme_y_values(arr, values=[0, 6]):
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

def find_edges(image, y_levels, values=[0, 6], min_width=19):
    edges_dict = {}
    for y in y_levels:
        row = image[y, :]
        mask = np.isin(row, values).astype(int)
        padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
        diff = np.diff(padded_mask)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        filtered_edges = [(start, end) for start, end in zip(starts, ends) if end - start + 1 >= min_width and 0 not in (start, end) and image.shape[1]-1 not in (start, end)]
        if filtered_edges:
            edges_dict[y] = filtered_edges
    return filter_crossings(image, edges_dict)

def find_rails(arr, y_levels, values=[9, 10], min_width=5):
    edges_all = []
    for y in y_levels:
        row = arr[y, :]
        mask = np.isin(row, values).astype(int)
        padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
        diff = np.diff(padded_mask)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        filtered_edges = [(start, end) for start, end in zip(starts, ends) if end - start + 1 >= min_width and 0 not in (start, end) and arr.shape[1]-1 not in (start, end)]
        edges_all.extend(filtered_edges)
    return edges_all

def robust_rail_sides(border, threshold=7):
    border = np.array(border)
    if border.size < 4: return border
    border = border[border[:, 1] != 1079]
    if border.size < 4: return border
    steps_x = np.diff(border[:, 0])
    median_step = np.median(np.abs(steps_x))
    threshold_step = np.abs(threshold * np.abs(median_step))
    if threshold_step == 0: threshold_step = 20
    treshold_overcommings = abs(steps_x) > abs(threshold_step)
    if not np.any(treshold_overcommings): return border
    overcommings_indices = np.where(treshold_overcommings)[0]
    split_indices = np.concatenate(([0], overcommings_indices + 1, [len(border)]))
    best_segment = max([border[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)], key=len)
    return best_segment

def find_rail_sides(img, edges_dict):
    left_border, right_border = [], []
    for y, xs in edges_dict.items():
        if not xs: continue
        rails = find_rails(img, [y], values=[9,10], min_width=5)
        left_border_actual = [min(xs)[0], y]
        right_border_actual = [max(xs)[1], y]
        for zone in rails:
            if abs(zone[1] - left_border_actual[0]) < y * 0.04: left_border_actual[0] = zone[0]
            if abs(zone[0] - right_border_actual[0]) < y * 0.04: right_border_actual[0] = zone[1]
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

def manage_detections(results):
    # ultralytics 결과 객체에서 직접 필터링
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
    colors_bgr = [(0, 255, 255), (0, 165, 255), (0, 0, 255)] # Yellow, Orange, Red
    safe_color = (0, 255, 0) # Green

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
            "xywh": box.xywh[0].cpu().numpy().tolist(), "conf": box.conf[0].cpu().numpy().item(), 
            "cls_name": names[int(box.cls)], "color": color
        })
    return boxes_info

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
# -------------------------------------------------------------------
# 추론 및 시각화 함수
# -------------------------------------------------------------------
# process 함수를 스크립트 내로 가져와 의존성 제거
def process_segformer(model, input_img, output_size):
    seg_logits = model(input_img).logits
    upsampled_logits = torch.nn.functional.interpolate(
        seg_logits,
        size=output_size,
        mode="bilinear",
        align_corners=False
    )
    id_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return id_map

def run_inference_and_draw(frame, model_seg, model_det, transform_seg, target_distances, device, num_ys=15):
    """단일 프레임에 대해 추론과 시각화를 모두 수행합니다."""
    
    # 1. 추론
    with torch.no_grad():
        with autocast(enabled=(device.type == 'cuda')):
            # SegFormer 추론 (PyTorch FP16)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor_seg = transform_seg(image=frame_rgb)['image'].unsqueeze(0).to(device)
            original_h, original_w, _ = frame.shape
            id_map = process_segformer(model_seg, input_tensor_seg, (original_h, original_w))
            
            # 후처리
            clues = get_clues(id_map, num_ys)
            edges = find_edges(id_map, clues, min_width=int(original_w * 0.01))
            ego_edges = identify_ego_track(edges, id_map.shape[1])
            borders = border_handler(id_map, ego_edges, target_distances)
            
            # YOLO 추론 (TensorRT 엔진)
            results = model_det(frame, verbose=False) 
            boxes_moving = manage_detections(results)
            classification = classify_detections(boxes_moving, borders, results[0].names)

    # 2. 시각화
    overlay = frame.copy()
    alpha = 0.2
    colors_bgr = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]

    for i, border_pair in enumerate(reversed(borders)):
        border_l, border_r = np.array(border_pair[0], dtype=np.int32), np.array(border_pair[1], dtype=np.int32)
        if border_l.size > 0 and border_r.size > 0:
            poly_points = np.concatenate((border_l, border_r[::-1]), axis=0)
            cv2.fillPoly(overlay, [poly_points], colors_bgr[i])

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, border_pair in enumerate(reversed(borders)):
        border_l, border_r = np.array(border_pair[0], dtype=np.int32), np.array(border_pair[1], dtype=np.int32)
        if border_l.size > 0: cv2.polylines(frame, [border_l], isClosed=False, color=colors_bgr[i], thickness=2)
        if border_r.size > 0: cv2.polylines(frame, [border_r], isClosed=False, color=colors_bgr[i], thickness=2)

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

    parser = argparse.ArgumentParser(description="RailSafeNet Hybrid Performance Demo.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--seg_width", type=int, default=480)
    parser.add_argument("--seg_height", type=int, default=270)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    PATH_model_seg = '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth'
    PATH_model_det_engine = '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.engine'
    
    # 1. SegFormer는 PyTorch 모델로 로드하고 FP16으로 변환
    model_seg = load_model(PATH_model_seg).to(device).eval()
    if device.type == 'cuda':
        model_seg.half() # FP16으로 변환
        print("SegFormer model converted to FP16.")

    # 2. YOLO는 최적화된 TensorRT 엔진으로 로드
    if not os.path.exists(PATH_model_det_engine):
        print(f"Error: YOLO TensorRT engine not found at {PATH_model_det_engine}")
        print("Please run 'yolo export model=... format=tensorrt half=True' first.")
        exit()
    print(f"Loading YOLO TensorRT engine from: {PATH_model_det_engine}")
    model_det = YOLO(PATH_model_det_engine)
    
    print("Models loaded successfully.")
    
    transform_seg = A.Compose([
        A.Resize(height=args.seg_height, width=args.seg_width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    target_distances = [650, 1000, 2000]

    cap = cv2.VideoCapture(args.input)
    fps_deque = deque(maxlen=30)
    
    # "warm up" a few frames
    for _ in range(5):
      ret, frame = cap.read()
      if ret:
        _ = run_inference_and_draw(frame, model_seg, model_det, transform_seg, target_distances, device)
    
    print("Warm-up complete. Starting main processing loop...")
    
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = run_inference_and_draw(frame, model_seg, model_det, transform_seg, target_distances, device)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_deque.append(fps)
        avg_fps = sum(fps_deque) / len(fps_deque)
        
        cv2.putText(processed_frame, f"Processing FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('RailSafeNet Hybrid Performance', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()