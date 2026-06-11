import torch
import cv2
import os
import numpy as np
import gc
import time
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import torchvision.transforms as T
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import scipy.linalg  # <--- BUG 2 FIXED: Added this import for the Kalman Filter

# ==========================================
# 1. PHYSICS ENGINE (Kalman Filter)
# ==========================================
class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean = np.dot(self._update_mat, mean)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + np.diag(np.square(std))
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

def xyxy_to_xyah(bbox):
    ret = np.zeros(4)
    ret[0] = (bbox[0] + bbox[2]) / 2.
    ret[1] = (bbox[1] + bbox[3]) / 2.
    ret[3] = bbox[3] - bbox[1]
    ret[2] = (bbox[2] - bbox[0]) / ret[3]
    return ret

# ==========================================
# 2. DEEPSORT MATCHER (Physics + Appearance)
# ==========================================
class DeepSORTMatcher:
    def __init__(self, max_cosine_distance=0.2, max_age=30):
        self.max_distance = max_cosine_distance
        self.max_age = max_age
        self.next_id = 1
        self.kf = KalmanFilter() 
        self.tracks = {}

    def update(self, bboxes, embeddings):
        if len(bboxes) == 0:
            self._age_tracks()
            return []

        # 1. Physics Prediction
        for t_id in self.tracks.keys():
            mean, cov = self.tracks[t_id]['mean'], self.tracks[t_id]['cov']
            self.tracks[t_id]['mean'], self.tracks[t_id]['cov'] = self.kf.predict(mean, cov)

        if len(self.tracks) == 0:
            return self._init_new_tracks(bboxes, embeddings)

        # 2. Match based on Appearance (Cosine Distance)
        track_ids = list(self.tracks.keys())
        track_embeddings = np.array([self.tracks[tid]['embedding'] for tid in track_ids])
        cost_matrix = 1.0 - np.dot(track_embeddings, embeddings.T)

        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        matches, unmatched_tracks, unmatched_dets = [], set(range(len(track_ids))), set(range(len(bboxes)))

        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] <= self.max_distance:
                matches.append((t_idx, d_idx))
                unmatched_tracks.discard(t_idx)
                unmatched_dets.discard(d_idx)

        # 3. Update Memory
        assigned_ids = [None] * len(bboxes)
        for t_idx, d_idx in matches:
            t_id = track_ids[t_idx]
            assigned_ids[d_idx] = t_id
            
            old_emb = self.tracks[t_id]['embedding']
            merged_emb = 0.9 * old_emb + 0.1 * embeddings[d_idx]
            self.tracks[t_id]['embedding'] = merged_emb / np.linalg.norm(merged_emb)
            
            measurement = xyxy_to_xyah(bboxes[d_idx])
            mean, cov = self.tracks[t_id]['mean'], self.tracks[t_id]['cov']
            self.tracks[t_id]['mean'], self.tracks[t_id]['cov'] = self.kf.update(mean, cov, measurement)
            
            self.tracks[t_id]['bbox'] = bboxes[d_idx] 
            self.tracks[t_id]['time_lost'] = 0 

        for d_idx in unmatched_dets:
            self._register_track(bboxes[d_idx], embeddings[d_idx], d_idx, assigned_ids)
        for t_idx in unmatched_tracks:
            self.tracks[track_ids[t_idx]]['time_lost'] += 1

        dead_tracks = [t_id for t_id, data in self.tracks.items() if data['time_lost'] > self.max_age]
        for t_id in dead_tracks: del self.tracks[t_id]

        return assigned_ids

    def _init_new_tracks(self, bboxes, embeddings):
        assigned_ids = [None] * len(bboxes)
        for i in range(len(bboxes)):
            self._register_track(bboxes[i], embeddings[i], i, assigned_ids)
        return assigned_ids

    def _register_track(self, bbox, embedding, idx, assigned_ids_list=None):
        mean, cov = self.kf.initiate(xyxy_to_xyah(bbox))
        self.tracks[self.next_id] = {
            'embedding': embedding,
            'bbox': bbox,
            'mean': mean,
            'cov': cov,
            'time_lost': 0
        }
        if assigned_ids_list is not None:
            assigned_ids_list[idx] = self.next_id
        self.next_id += 1

    def _age_tracks(self):
        for t_id in list(self.tracks.keys()):
            self.tracks[t_id]['time_lost'] += 1
            if self.tracks[t_id]['time_lost'] > self.max_age: del self.tracks[t_id]

# ==========================================
# 3. FEATURE EXTRACTOR MODULE (ReID)
# ==========================================
class DeepSORTFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity() 
        self.model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)), 
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, frame, bboxes):
        if len(bboxes) == 0: return None, []
        crops, valid_indices = [], []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(self.transform(crop))
                valid_indices.append(i)
        
        if not crops: return None, []

        input_tensor = torch.stack(crops).to(self.device)
        with torch.no_grad():
            embeddings = self.model(input_tensor)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().numpy(), bboxes[valid_indices]

# ==========================================
# 4. SETUP & PATHS
# ==========================================
video_path = 'test_vedio.mp4'
output_path = 'navya_final_optimized_pipeline.mp4'
if not os.path.exists(video_path): 
    print(f"Error: {video_path} not found.")
    exit()
  
print("Loading Models onto RTX 3050...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", torch_dtype=torch.float16).to("cuda")
yolo_model = YOLO('yolo11s-seg.pt')

reid_extractor = DeepSORTFeatureExtractor(device='cuda')
tracker = DeepSORTMatcher(max_cosine_distance=0.2, max_age=30)

cap = cv2.VideoCapture(video_path)
orig_width, orig_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
TARGET_PROCESSING_HEIGHT = 1024 
final_height = min(orig_height, TARGET_PROCESSING_HEIGHT)
final_width = int(final_height * (orig_width / orig_height))
LEGEND_WIDTH = 300
COMBINED_WIDTH = final_width + LEGEND_WIDTH
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (COMBINED_WIDTH, final_height))

palette_bgr = np.array([[128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152], [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119]], dtype=np.uint8)
class_names = ["Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"]

processing_times = []

# ==========================================
# 5. FRAME-BY-FRAME LOOP (With TTC Calculation)
# ==========================================
with torch.no_grad():
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # Process the ENTIRE frame directly
        processed_frame = cv2.resize(frame, (final_width, final_height))
        start_time = time.time()

        # --- A. YOLO INSTANCE INFERENCE ---
        yolo_results = yolo_model.track(
            source=processed_frame, persist=True, tracker="botsort.yaml", 
            conf=0.35, iou=0.45, imgsz=832, retina_masks=True, verbose=False
        )

        # --- B. SEGFORMER SEMANTIC INFERENCE ---
        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda", dtype=torch.float16)
        outputs = segformer_model(**inputs)
        
        # Interpolate across the FULL height and width
        prediction = torch.nn.functional.interpolate(
            outputs.logits.to(torch.float32), size=(final_height, final_width), 
            mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        # --- C. BASE LAYERING ---
        seg_color_map = palette_bgr[prediction]
        layered_frame = cv2.addWeighted(processed_frame, 0.5, seg_color_map, 0.5, 0)

        # --- D. FEATURE EXTRACTION, DEEPSORT IDs, & TTC ---
        if yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            embeddings, valid_boxes = reid_extractor.extract(processed_frame, boxes)
            
            if embeddings is not None:
                assigned_ids = tracker.update(valid_boxes, embeddings)
                
                box_color = (255, 255, 0) # Cyan for normal tracking
                
                for i, box in enumerate(valid_boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    t_id = assigned_ids[i]
                    
                    # Draw base tracking box
                    cv2.rectangle(layered_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(layered_frame, f"ID: {t_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                    # --- TTC CALCULATION LOGIC ---
                    track_data = tracker.tracks.get(t_id)
                    if track_data is not None:
                        # Extract Kalman states: height (h) and height velocity (vh)
                        cx, cy, a, h = track_data['mean'][:4]
                        vx, vy, va, vh = track_data['mean'][4:]

                        # Define "Ego Vehicle Path" as the middle 50% of the screen
                        in_ego_path = (final_width * 0.25) < cx < (final_width * 0.75)

                        # If in path and bounding box is expanding fast enough (vh > 0.5 to filter noise)
                        if in_ego_path and vh > 0.5:
                            ttc = h / vh
                            
                            # Trigger High-Priority Warning
                            if ttc < 2.0:
                                print(f"WARNING: Object ID {t_id} TTC < 2.0s ({ttc:.2f}s)!")
                                
                                # Visual UI Alert (Changes Box to Red and flashes Warning)
                                cv2.rectangle(layered_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(layered_frame, f"WARNING: TTC {ttc:.1f}s", (x1, max(20, y1 - 30)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        end_time = time.time()
        processing_times.append(end_time - start_time)

        # --- E. RECONSTRUCT FULL FRAME & UI ---
        combined_output = np.full((final_height, COMBINED_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
        combined_output[:, :final_width] = layered_frame 

        for i, name in enumerate(class_names):
            y_pos = 90 + (i * 30)
            cv2.rectangle(combined_output, (final_width + 15, y_pos - 15), (final_width + 45, y_pos + 5), palette_bgr[i].tolist(), -1)
            cv2.putText(combined_output, name, (final_width + 60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(combined_output, f"FPS: {1.0 / (end_time - start_time):.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_output, f"Physics Tracking & TTC Active", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(combined_output)


# ==========================================
# 6. CLEANUP & PROFILING 
# ==========================================
cap.release()
out.release()

avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
print("\n" + "="*50)
print(" PERCEPTION TEAM: FINAL DELIVERABLE REPORT")
print("="*50)
print(f"Total Frames Processed: {total_frames}")
print(f"Average Pipeline Speed: {1.0 / avg_time if avg_time > 0 else 0:.2f} FPS")
print(f"Video Saved To: {output_path}")
print("="*50)

print("Clearing VRAM...")
try:
    del segformer_model, processor, yolo_model, reid_extractor
except NameError:
    pass 
gc.collect() 
torch.cuda.empty_cache() 
print("VRAM cleared. GPU is now cool and quiet. Safe to exit.")