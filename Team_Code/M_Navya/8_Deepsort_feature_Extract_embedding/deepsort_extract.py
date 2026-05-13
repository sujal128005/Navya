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

# ==========================================
# 1. NEW: FEATURE EXTRACTOR MODULE (ReID)
# ==========================================
class DeepSORTFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        # Using a standard ResNet backbone for ReID (common in DeepSORT)
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove classifier to get 512-d embeddings
        self.model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),  # Standard DeepSORT input size
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, frame, bboxes):
        """
        Outputs: 
        - features: torch.Tensor of shape (N, 512)
        - valid_bboxes: The bboxes that were successfully processed
        """
        if len(bboxes) == 0:
            return None, []

        crops = []
        valid_indices = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box[:4])
            # Clip coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(self.transform(crop))
                valid_indices.append(i)
        
        if not crops:
            return None, []

        input_tensor = torch.stack(crops).to(self.device)
        with torch.no_grad():
            # Calculate embeddings and normalize for Cosine Distance
            embeddings = self.model(input_tensor)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings, bboxes[valid_indices]

# ==========================================
# 2. SETUP & PATHS (Same as your code)
# ==========================================
video_path = 'test_vedio.mp4'
output_path = 'navya_final_optimized_pipeline.mp4'
if not os.path.exists(video_path): exit()

# ==========================================
# 3. LOAD MODELS (Including Extractor)
# ==========================================
print("Loading Models onto RTX 3050...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", torch_dtype=torch.float16).to("cuda")
yolo_model = YOLO('yolo11s-seg.pt')

# Initialize our new Extractor
reid_extractor = DeepSORTFeatureExtractor(device='cuda')

# (Video Capture and Dimension logic remains the same as your provided code)
cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
TARGET_PROCESSING_HEIGHT = 1024 
final_height = min(orig_height, TARGET_PROCESSING_HEIGHT)
final_width = int(final_height * (orig_width / orig_height))
LEGEND_WIDTH = 300
COMBINED_WIDTH = final_width + LEGEND_WIDTH
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (COMBINED_WIDTH, final_height))
dynamic_y_start = int(final_height * 0.25)
sky_class_idx = 10 
sky_buffer = 40 

# (Palette and Class Names remain the same)
palette_bgr = np.array([[128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152], [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119]], dtype=np.uint8)
class_names = ["Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"]

processing_times = []

# ==========================================
# 4. FRAME-BY-FRAME LOOP
# ==========================================
with torch.no_grad():
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        processed_frame = cv2.resize(frame, (final_width, final_height))
        current_y_start = max(0, min(dynamic_y_start, final_height - 100))
        roi_height = final_height - current_y_start
        roi_frame = processed_frame[current_y_start:final_height, :]

        start_time = time.time()

        # --- A. YOLO INSTANCE INFERENCE ---
        yolo_results = yolo_model.track(
            source=roi_frame, persist=True, tracker="botsort.yaml", 
            conf=0.35, iou=0.45, imgsz=832, retina_masks=True, verbose=False
        )

        # --- B. NEW: FEATURE EXTRACTION ---
        # We extract boxes from YOLO and feed them to our ReID extractor
        if yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            embeddings, valid_boxes = reid_extractor.extract(roi_frame, boxes)
            # 'embeddings' is your tensor deliverable for DeepSORT
        
        # --- C. SEGFORMER SEMANTIC INFERENCE ---
        image_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda", dtype=torch.float16)
        outputs = segformer_model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.logits.to(torch.float32), size=(roi_height, final_width), 
            mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        # --- D. DYNAMIC HORIZON & LAYERING ---
        sky_pixels = np.where(prediction == sky_class_idx)
        if len(sky_pixels[0]) > 0:
            absolute_sky_y = current_y_start + np.max(sky_pixels[0])
            dynamic_y_start = max(0, absolute_sky_y - sky_buffer)
        else:
            dynamic_y_start = max(0, current_y_start - 30)

        seg_color_map = palette_bgr[prediction]
        semantic_overlay_roi = cv2.addWeighted(roi_frame, 0.5, seg_color_map, 0.5, 0)
        yolo_results[0].orig_img = semantic_overlay_roi
        layered_roi = yolo_results[0].plot()

        end_time = time.time()
        processing_times.append(end_time - start_time)

        # --- E. RECONSTRUCT & LEGEND ---
        final_layered_frame = processed_frame.copy()
        final_layered_frame[current_y_start:final_height, :] = layered_roi

        combined_output = np.full((final_height, COMBINED_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
        combined_output[:, :final_width] = final_layered_frame

        # Drawing Legend (Simplified for brevity)
        for i, name in enumerate(class_names):
            y_pos = 90 + (i * 30)
            cv2.rectangle(combined_output, (final_width + 15, y_pos - 15), (final_width + 45, y_pos + 5), palette_bgr[i].tolist(), -1)
            cv2.putText(combined_output, name, (final_width + 60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Telemetry Display
        current_fps = 1.0 / (end_time - start_time)
        cv2.putText(combined_output, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_output, f"Embeddings: {len(embeddings) if embeddings is not None else 0}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(combined_output)

# ==========================================
# 7. CLEANUP & PROFILING REPORT (Optimized)
# ==========================================
cap.release()
out.release()

avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
avg_fps = 1.0 / avg_time if avg_time > 0 else 0

print("\n" + "="*50)
print(" PERCEPTION TEAM: FINAL DELIVERABLE REPORT")
print("="*50)
print(f"Total Frames Processed: {total_frames}")
print(f"Average Pipeline Speed: {avg_fps:.2f} FPS")
print(f"Video Saved To: {output_path}")
print("="*50)

# --- THE VRAM WIPE ---
# Explicitly delete the new ReID module alongside the others
print("Clearing VRAM...")

try:
    del segformer_model
    del processor
    del yolo_model
    del reid_extractor  # <--- Added this to clear the DeepSORT features module
except NameError:
    pass 

gc.collect() 
torch.cuda.empty_cache() 

print("VRAM cleared. GPU is now cool and quiet. Safe to exit.")