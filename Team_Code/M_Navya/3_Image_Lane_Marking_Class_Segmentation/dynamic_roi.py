import torch
import cv2
import os
import numpy as np
import gc
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
# Define input and output paths for clarity
img_path = 'test_road.jpg'
mask_index_path = 'mask_indices.png'  # Grayscale image of raw class indices
mask_color_path = 'segmentation_map.jpg'  # Color-coded segmentation map
overlay_path = 'overlay_output.jpg'     # Image blended with the color map

# Functional Description: Check if input image exists
if not os.path.exists(img_path):
    print(f"Input file not found: {img_path}")
    exit()

# Functional Description: Load the heavy SegFormer model for high-accuracy segmentation
# This model uses the Cityscapes dataset, which labels 19 distinct classes.
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
print(f"Loading SegFormer model: {model_name}... This may take a moment.")
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to("cuda")

# ==========================================
# 2. CORE SEGMENTATION INFERENCE
# ==========================================
# Read input image
image_bgr = cv2.imread(img_path)
height, width = image_bgr.shape[:2]

# Functional Description: Preprocess image for the model and run inference on GPU
# SegFormer requires input in RGB format.
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
inputs = processor(images=image_rgb, return_tensors="pt").to("cuda")

print("Processing semantic segmentation...")
with torch.no_grad():
    outputs = model(**inputs)
    
    # Functional Description: Resize model output to match original image size
    # logits are interpolated from 256x256 to original, then argmax finds the dominant class per pixel.
    # The resulting `prediction` is a numpy array of indices (0-18) for each pixel.
    prediction = torch.nn.functional.interpolate(
        outputs.logits, size=(height, width), mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

# ==========================================
# 3. COLOR MAPPING & BLENDING
# ==========================================
# Functional Description: Efficiently map each predicted class index to a unique color tuple
# Cityscapes palette (BGR colors)
palette_bgr = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [153, 153, 190], # 0-4
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152], # 5-9
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],              # 10-14
    [100, 60, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119]                             # 15-18
], dtype=np.uint8)

# Generate a color-coded segmentation map with distinct colors per class
# (e.g., Road is purple, cars are blue, sky is cyan)
seg_color_mask = palette_bgr[prediction]

# Functional Description: Blend the input image with the color segmentation map (50% blend)
# The legend drawing logic is removed here to ensure a simple and clean result.
blended_overlay = cv2.addWeighted(image_bgr, 0.5, seg_color_mask, 0.5, 0)

# ==========================================
# 4. SAVE RESULTS & CLEANUP VRAM
# ==========================================
# 1. Save raw class indices (values 0-18 for Cityscapes classes)
visible_mask = (prediction * (255 / 18)).astype(np.uint8)
cv2.imwrite(mask_index_path, visible_mask)

# 2. Save the pure, color-coded segmentation map
cv2.imwrite(mask_color_path, seg_color_mask)

# 3. Save the clean, blended overlay with no unnecessary telemetry or text
cv2.imwrite(overlay_path, blended_overlay)

print(f"DONE! Please check for:")
print(f"1. Raw class indices:   '{mask_index_path}'")
print(f"2. Color segmentation map: '{mask_color_path}'")
print(f"3. Blended image-map:   '{overlay_path}'")

# Functional Description: Explicitly delete large model objects to free up GPU memory
del model
del processor
gc.collect() # run garbage collection
torch.cuda.empty_cache() # clear the PyTorch VRAM cache
print("GPU memory (VRAM) has been cleared.")