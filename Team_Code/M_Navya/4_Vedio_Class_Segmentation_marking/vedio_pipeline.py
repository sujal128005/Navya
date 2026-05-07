import torch
import cv2
import os
import numpy as np
import gc
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# 1. SETUP & PATHS
# ==========================================
video_path = 'test_vedio.mp4'  # Replace with your input video file
output_path = 'segmented_output_final.mp4'

if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit()

# ==========================================
# 2. LOAD HIGH-ACCURACY MODEL
# ==========================================
# Re-confirming we are using the heavy, high-accuracy B5 checkpoint
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
print(f"Loading top-tier accuracy model: {model_name}... (Expect significant memory usage)")
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to("cuda")

# ==========================================
# 3. VIDEO CAPTURE & DIMENSION SETTINGS
# ==========================================
cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# define target height for processing and final output
# this keeps inference feasible without blowing VRAM
TARGET_PROCESSING_HEIGHT = 1024 

if orig_height > TARGET_PROCESSING_HEIGHT:
    aspect_ratio = orig_width / orig_height
    final_height = TARGET_PROCESSING_HEIGHT
    final_width = int(final_height * aspect_ratio)
else:
    final_height = orig_height
    final_width = orig_width

# Define dimensions for the dedicated side legend panel
LEGEND_WIDTH = 300
COMBINED_WIDTH = final_width + LEGEND_WIDTH

# Define output VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # use mp4v for high compatibility
out = cv2.VideoWriter(output_path, fourcc, fps, (COMBINED_WIDTH, final_height))

# ==========================================
# 4. PALETTE INFORMATION (CITYSCAPES)
# ==========================================
# Standard distinct BGR color tuples for all 19 classes
# (This ensures the output is standard and highly visually distinct)
palette_bgr = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

# Map indices to human-readable names
class_names = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
    "Sky", "Person", "Rider", "Car", "Truck",
    "Bus", "Train", "Motorcycle", "Bicycle"
]

print(f"Processing Video: {total_frames} frames.")
print(f"Original Resolution: {orig_width}x{orig_height}")
print(f"Processing Resolution: {final_width}x{final_height}")
print(f"Final Resolution (with legend): {COMBINED_WIDTH}x{final_height}")

# ==========================================
# 5. FRAME-BY-FRAME LOOP (AI PROCESSING)
# ==========================================
# use torch.no_grad to disable gradient calculation for efficiency
with torch.no_grad():
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to our targeted processing height while maintaining aspect ratio
        processed_frame = cv2.resize(frame, (final_width, final_height))

        # --- PREPROCESSING ---
        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        # Preprocess input image (resize, normalize, and move to CUDA for high-accuracy inference)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda")

        # --- AI INFERENCE ---
        # Generate segmentation logits
        outputs = model(**inputs)
        # Resize logits to match the processed resolution
        prediction = torch.nn.functional.interpolate(
            outputs.logits, size=(final_height, final_width), mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        # --- MAP SEGMENTATION TO COLORS ---
        # efficiently map each predicted class index to its corresponding distinct color tuple
        seg_color_map = palette_bgr[prediction]

        # blend the segmentation color map with the resized original frame (50% transparency)
        overlay = cv2.addWeighted(processed_frame, 0.5, seg_color_map, 0.5, 0)

        # ==========================================
        # 6. BUILD DEDICATED LEGEND PANEL
        # ==========================================
        # Create a new BGR frame initialized with a distinct background color (e.g., neutral dark grey)
        combined_output = np.full((final_height, COMBINED_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
        # place the overlay video content on the left side
        combined_output[:, :final_width] = overlay

        # Draw the legend into the dedicated panel (right side)
        legend_start_y = 50
        cv2.putText(combined_output, "CITYSCAPES LEGEND", (final_width + 10, legend_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        legend_y = legend_start_y + 40
        for class_idx in range(len(palette_bgr)):
            class_bgr_color = palette_bgr[class_idx]
            color_tuple = (int(class_bgr_color[0]), int(class_bgr_color[1]), int(class_bgr_color[2]))

            # Functional Description: draw swatches with the associated class color and text label next to it
            cv2.rectangle(combined_output, (final_width + 15, legend_y - 15), (final_width + 45, legend_y + 5), color_tuple, -1)
            cv2.putText(combined_output, class_names[class_idx], (final_width + 60, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 30

        # write the wider combined frame to the final video
        out.write(combined_output)

# ==========================================
# 7. CLEANUP & VRAM CLENSING (Prioritized)
# ==========================================
# explicitly clean up all resources
cap.release()
out.release()
print(f"Processing complete! Saved to {output_path}")

# Explicitly delete all model references to free large memory buffers
del model
del processor
gc.collect() # run garbage collection
torch.cuda.empty_cache() # clear the PyTorch VRAM cache
print("GPU memory (VRAM) has been completely cleared.")