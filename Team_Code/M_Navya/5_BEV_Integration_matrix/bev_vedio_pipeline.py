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
video_path = 'test_drive2.mp4'  # Replace with your input video file
output_path = 'segmented_bev_output_final.mp4'

if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit()

# ==========================================
# 2. LOAD HIGH-ACCURACY MODEL
# ==========================================
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

TARGET_PROCESSING_HEIGHT = 1024 

if orig_height > TARGET_PROCESSING_HEIGHT:
    aspect_ratio = orig_width / orig_height
    final_height = TARGET_PROCESSING_HEIGHT
    final_width = int(final_height * aspect_ratio)
else:
    final_height = orig_height
    final_width = orig_width

# Define dimensions for output panels
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
print(f"Original Resolution: {orig_width}x{orig_height}")
print(f"Processing Resolution: {final_width}x{final_height}")

# ==========================================
# 5. FRAME-BY-FRAME LOOP (AI PROCESSING)
# ==========================================
with torch.no_grad():
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = cv2.resize(frame, (final_width, final_height))

        # --- PREPROCESSING ---
        image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda")

        # --- AI INFERENCE ---
        outputs = model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.logits, size=(final_height, final_width), mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        # --- MAP SEGMENTATION TO COLORS ---
        seg_color_map = palette_bgr[prediction]
        overlay = cv2.addWeighted(processed_frame, 0.5, seg_color_map, 0.5, 0)

        # ==========================================
        # 6. BEV TRANSFORM & POLYGON EXTRACTION
        # ==========================================
        # Isolate the drivable area mask (Class 0 = Road)
        drivable_mask = np.uint8(prediction == 0) * 255

        # --- YOUR TUNED SOURCE POINTS ---
        h, w = final_height, final_width
        src_points = np.float32([
            [w * 0.0000, h * 0.5200],
            [w * 1.0000, h * 0.5200],
            [w * 0.0100, h * 0.5800],
            [w * 0.9900, h * 0.5800]
        ])

        # Define the size of the output BEV panel
        bev_width, bev_height = 400,400
        dst_points = np.float32([
            [0, 0],                     
            [bev_width, 0],             
            [0, bev_height],            
            [bev_width, bev_height]     
        ])

        # Calculate the Homography Matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 1. Warp the binary mask to get the mathematical polygon boundaries
        bev_mask = cv2.warpPerspective(drivable_mask, matrix, (bev_width, bev_height), flags=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(bev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 2. Warp the ACTUAL VIDEO FRAME so you can see the road, lanes, and texture
        bev_color_output = cv2.warpPerspective(processed_frame, matrix, (bev_width, bev_height))
        
        # 3. Draw the green polygon on top of the real road image
        cv2.drawContours(bev_color_output, contours, -1, (0, 255, 0), 2)

        # Display the BEV live 
        cv2.imshow("Live BEV Telemetry", bev_color_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # ==========================================
        # 7. BUILD DEDICATED LEGEND PANEL
        # ==========================================
        combined_output = np.full((final_height, COMBINED_WIDTH, 3), (40, 40, 40), dtype=np.uint8)
        combined_output[:, :final_width] = overlay

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
# 8. CLEANUP & VRAM CLENSING 
# ==========================================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete! Saved to {output_path}")

del model
del processor
gc.collect() 
torch.cuda.empty_cache() 
print("GPU memory (VRAM) has been completely cleared.")