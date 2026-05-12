import torch
import cv2
import os
import numpy as np
import gc
import time
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

# ==========================================
# 1. SETUP & PATHS
# ==========================================
video_path = 'test_drive2.mp4'  # Using your exact filename
output_path = 'navya_final_optimized_pipeline.mp4'

if not os.path.exists(video_path):
    print(f"Error: File not found: {video_path}")
    exit()

# ==========================================
# 2. LOAD HIGH-ACCURACY MODELS
# ==========================================
# SegFormer (Semantic: Roads, Buildings)
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
print("Loading Semantic Model (SegFormer)...")
processor = SegformerImageProcessor.from_pretrained(model_name)
segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# YOLOv11 (Instance/Tracking: Overlapping Vehicles)
print("Loading Instance Tracking Model (YOLO11)...")
yolo_model = YOLO('yolo11s-seg.pt')

# ==========================================
# 3. VIDEO CAPTURE & DIMENSION SETTINGS
# ==========================================
cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

TARGET_PROCESSING_HEIGHT = 1024 

if orig_height > TARGET_PROCESSING_HEIGHT:
    aspect_ratio = orig_width / orig_height
    final_height = TARGET_PROCESSING_HEIGHT
    final_width = int(final_height * aspect_ratio)
else:
    final_height = orig_height
    final_width = orig_width

LEGEND_WIDTH = 300
COMBINED_WIDTH = final_width + LEGEND_WIDTH

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (COMBINED_WIDTH, final_height))

# --- DYNAMIC HORIZON TRACKER SETUP ---
dynamic_y_start = int(final_height * 0.25) # Start by dropping top 25% on Frame 1
sky_class_idx = 10 # Cityscapes ID for 'Sky'
sky_buffer = 40    # Keep 40 pixels of sky visible for the next frame

# ==========================================
# 4. PALETTE INFORMATION (CITYSCAPES)
# ==========================================
palette_bgr = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

class_names = [
    "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", 
    "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", 
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
]

processing_times = []
print(f"Processing Video: {total_frames} frames.")

# ==========================================
# 5. FRAME-BY-FRAME LOOP (AI PROCESSING)
# ==========================================
with torch.no_grad():
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        processed_frame = cv2.resize(frame, (final_width, final_height))

        processed_frame = cv2.resize(frame, (final_width, final_height))

        # --- PREPARE DYNAMIC ROI ---
        # 1. Lock in the Y-start for THIS specific frame
        current_y_start = max(0, min(dynamic_y_start, final_height - 100))
        roi_height = final_height - current_y_start
        roi_frame = processed_frame[current_y_start:final_height, :]

        # [ START PROFILER TIMER ]
        start_time = time.time()

        # --- A. YOLO INSTANCE INFERENCE (On ROI) ---
        yolo_results = yolo_model.track(
            source=roi_frame, 
            persist=True, tracker="botsort.yaml", conf=0.35, iou=0.45, 
            imgsz=832, retina_masks=True, verbose=False
        )

        # --- B. SEGFORMER SEMANTIC INFERENCE (On ROI) ---
        image_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda", dtype=torch.float16)
        
        outputs = segformer_model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.logits.to(torch.float32), size=(roi_height, final_width), 
            mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        # --- C. DYNAMIC HORIZON MATH FOR NEXT FRAME ---
        sky_pixels = np.where(prediction == sky_class_idx)
        
        if len(sky_pixels[0]) > 0:
            lowest_sky_in_crop = np.max(sky_pixels[0])
            # Use current_y_start for the absolute math!
            absolute_sky_y = current_y_start + lowest_sky_in_crop
            # Update the global dynamic_y_start for the NEXT loop
            dynamic_y_start = max(0, absolute_sky_y - sky_buffer)
        else:
            dynamic_y_start = max(0, current_y_start - 30)

        # --- D. THE LAYERING TRICK ---
        seg_color_map = palette_bgr[prediction]
        semantic_overlay_roi = cv2.addWeighted(roi_frame, 0.5, seg_color_map, 0.5, 0)
        
        yolo_results[0].orig_img = semantic_overlay_roi
        layered_roi = yolo_results[0].plot()

        # [ END PROFILER TIMER ]
        end_time = time.time()
        processing_times.append(end_time - start_time)

        # --- E. RECONSTRUCT FULL FRAME ---
        final_layered_frame = processed_frame.copy()
        # 2. Paste the ROI back using the LOCKED current_y_start!
        final_layered_frame[current_y_start:final_height, :] = layered_roi


        # ==========================================
        # 6. BUILD DEDICATED LEGEND PANEL & OVERLAYS
        # ==========================================
        combined_output = np.full((final_height, COMBINED_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
        combined_output[:, :final_width] = final_layered_frame

        # Legend
        cv2.putText(combined_output, "CITYSCAPES LEGEND", (final_width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y = 90
        for class_idx in range(len(palette_bgr)):
            class_bgr_color = palette_bgr[class_idx]
            color_tuple = (int(class_bgr_color[0]), int(class_bgr_color[1]), int(class_bgr_color[2]))
            cv2.rectangle(combined_output, (final_width + 15, legend_y - 15), (final_width + 45, legend_y + 5), color_tuple, -1)
            cv2.putText(combined_output, class_names[class_idx], (final_width + 60, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 30

        # Telemetry Text Overlays
        current_fps = 1.0 / (end_time - start_time)
        crop_percent = (dynamic_y_start / final_height) * 100
        cv2.putText(combined_output, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_output, f"Dynamic Crop: {crop_percent:.1f}% (Top)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(combined_output)

# ==========================================
# 7. CLEANUP & PROFILING REPORT
# ==========================================
cap.release()
out.release()

avg_time = sum(processing_times) / len(processing_times)
avg_fps = 1.0 / avg_time

print("\n" + "="*50)
print(" PERCEPTION TEAM: FINAL DELIVERABLE REPORT")
print("="*50)
print(f"Total Frames Processed: {total_frames}")
print(f"Average Inference Time: {avg_time:.4f} seconds/frame")
print(f"Average Pipeline Speed: {avg_fps:.2f} FPS")
print(f"Video Saved To: {output_path}")
print("="*50)

# Memory Cleansing
del segformer_model
del processor
del yolo_model
gc.collect() 
torch.cuda.empty_cache() 
print("VRAM cleared. Safe to exit.")