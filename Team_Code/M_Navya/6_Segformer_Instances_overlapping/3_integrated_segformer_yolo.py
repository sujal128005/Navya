import torch
import cv2
import os
import numpy as np
import gc
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

# ==========================================
# 1. SETUP & PATHS
# ==========================================
video_path = 'test_vedio.mp4'  # Ensure correct spelling
output_path = 'navya_unified_perception.mp4'

if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit()

# ==========================================
# 2. LOAD HIGH-ACCURACY MODELS
# ==========================================
# SegFormer (Semantic: Roads, Buildings, Sky)
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
print(f"Loading Semantic Model: {model_name}...")
processor = SegformerImageProcessor.from_pretrained(model_name)
# Using torch.float16 (FP16) is highly recommended here to prevent VRAM crashes
segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# YOLOv11 (Instance/Tracking: Overlapping Cars, Pedestrians)
print("Loading Instance Tracking Model: YOLO11-Small...")
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
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
    "Sky", "Person", "Rider", "Car", "Truck",
    "Bus", "Train", "Motorcycle", "Bicycle"
]

print(f"Processing Video: {total_frames} frames.")

# ==========================================
# 5. FRAME-BY-FRAME LOOP (AI PROCESSING)
# ==========================================
with torch.no_grad():
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        processed_frame = cv2.resize(frame, (final_width, final_height))

        # --- A. YOLO INSTANCE INFERENCE ---
        # Run tracking on the CLEAN frame first
        yolo_results = yolo_model.track(
            source=processed_frame, 
            persist=True, 
            tracker="botsort.yaml", 
            conf=0.35, 
            iou=0.45, 
            retina_masks=True,
            verbose=False # Keeps terminal clean during tqdm loop
        )

        # --- B. SEGFORMER SEMANTIC INFERENCE ---
        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda", dtype=torch.float16)

        outputs = segformer_model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.logits.to(torch.float32), # Interpolate needs float32
            size=(final_height, final_width), 
            mode="bilinear", 
            align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        # --- C. THE LAYERING TRICK ---
        # 1. Build the Segformer background overlay
        seg_color_map = palette_bgr[prediction]
        semantic_overlay = cv2.addWeighted(processed_frame, 0.5, seg_color_map, 0.5, 0)

        # 2. Tell YOLO to draw its bounding boxes/masks on top of the semantic_overlay 
        # instead of the original frame
        yolo_results[0].orig_img = semantic_overlay
        final_layered_frame = yolo_results[0].plot()

        # ==========================================
        # 6. BUILD DEDICATED LEGEND PANEL
        # ==========================================
        combined_output = np.full((final_height, COMBINED_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
        
        # Place the fully layered video on the left
        combined_output[:, :final_width] = final_layered_frame

        # Draw the legend
        legend_start_y = 50
        cv2.putText(combined_output, "CITYSCAPES LEGEND", (final_width + 10, legend_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        legend_y = legend_start_y + 40
        for class_idx in range(len(palette_bgr)):
            class_bgr_color = palette_bgr[class_idx]
            color_tuple = (int(class_bgr_color[0]), int(class_bgr_color[1]), int(class_bgr_color[2]))

            cv2.rectangle(combined_output, (final_width + 15, legend_y - 15), (final_width + 45, legend_y + 5), color_tuple, -1)
            cv2.putText(combined_output, class_names[class_idx], (final_width + 60, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 30

        out.write(combined_output)

# ==========================================
# 7. CLEANUP & VRAM CLENSING
# ==========================================
cap.release()
out.release()

del segformer_model
del processor
del yolo_model
gc.collect() 
torch.cuda.empty_cache() 
print(f"Processing complete! Unified pipeline saved to {output_path}")
