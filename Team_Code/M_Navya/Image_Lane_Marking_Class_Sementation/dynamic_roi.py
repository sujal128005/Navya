import torch
import cv2
import os
import numpy as np
import gc
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. Path Check
img_path = 'test_road.jpg'
if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
    exit()

# 2. Load Model
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to("cuda")

# 3. Process Image
image = cv2.imread(img_path)
height, width = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
inputs = processor(images=image_rgb, return_tensors="pt").to("cuda")

print("AI is processing... please wait.")
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.nn.functional.interpolate(
        outputs.logits, size=(height, width), mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].cpu().numpy()

# ==========================================
# MULTI-CLASS COLOR MAPPING (Complete 19-Class BGR Palette)
# ==========================================
palette_info = {
    0:  ("Road", [128, 64, 128]),          # Purple
    1:  ("Sidewalk", [232, 35, 244]),      # Pink
    2:  ("Building", [70, 70, 70]),        # Dark Grey
    3:  ("Wall", [156, 102, 102]),         # Blue-Grey
    4:  ("Fence", [153, 153, 190]),        # Light Purple
    5:  ("Pole", [153, 153, 153]),         # Grey
    6:  ("Traffic Light", [30, 170, 250]), # Yellow/Orange
    7:  ("Traffic Sign", [0, 220, 220]),   # Cyan
    8:  ("Vegetation", [35, 142, 107]),    # Green
    9:  ("Terrain", [152, 251, 152]),      # Light Green
    10: ("Sky", [180, 130, 70]),           # Sky Blue
    11: ("Person", [60, 20, 220]),         # Red
    12: ("Rider", [0, 0, 255]),            # Bright Red
    13: ("Car", [142, 0, 0]),              # Dark Blue
    14: ("Truck", [70, 0, 0]),             # Navy Blue
    15: ("Bus", [100, 60, 0]),             # Teal
    16: ("Train", [100, 80, 0]),           # Dark Teal
    17: ("Motorcycle", [230, 0, 0]),       # Blue
    18: ("Bicycle", [32, 11, 119])         # Burgundy
}

seg_color_map = np.zeros_like(image)
for class_idx, (name, color) in palette_info.items():
    seg_color_map[prediction == class_idx] = color

overlay = cv2.addWeighted(image, 0.5, seg_color_map, 0.5, 0)

# ==========================================
# DRAW THE VISUAL LEGEND
# ==========================================
legend_x = width - 250  
legend_y = 40

cv2.putText(overlay, "CLASS LEGEND", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
legend_y += 30

for class_idx, (name, color) in palette_info.items():
    cv2.rectangle(overlay, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
    cv2.rectangle(overlay, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), (255, 255, 255), 1)
    cv2.putText(overlay, name, (legend_x + 35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25 

# ==========================================
# LANE DETECTION & TELEMETRY MATH
# ==========================================
road_mask = np.where(prediction == 0, 255, 0).astype(np.uint8)

kernel = np.ones((5, 5), np.uint8)
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

roi_top = int(height * 0.6)
road_mask[:roi_top, :] = 0

edges = cv2.Canny(road_mask, 100, 200)
edges[-20:, :] = 0 

midpoint = width // 2
left_lane_pixels = edges[:, :midpoint]
right_lane_pixels = edges[:, midpoint:]

left_y, left_x = np.nonzero(left_lane_pixels[roi_top:, :])
right_y, right_x = np.nonzero(right_lane_pixels[roi_top:, :])

left_y = left_y + roi_top
right_y = right_y + roi_top
right_x = right_x + midpoint

if len(left_x) > 50 and len(right_x) > 50:
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    y_eval = height
    left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    lane_center = (left_x_bottom + right_x_bottom) / 2
    vehicle_center = width / 2
    offset_pixels = vehicle_center - lane_center

    lane_width_pixels = right_x_bottom - left_x_bottom
    meters_per_pixel = 3.7 / lane_width_pixels
    center_offset_meters = offset_pixels * meters_per_pixel

    direction = "Right" if center_offset_meters > 0 else "Left"
    
    xm_per_pix = meters_per_pixel 
    ym_per_pix = 30.0 / height

    left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
    y_eval_real = height * ym_per_pix

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_real + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_real + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    road_curve_radius = (left_curverad + right_curverad) / 2

    if road_curve_radius > 3000:
        curve_text = "The road is essentially STRAIGHT."
    else:
        curve_text = f"Road Curve Radius: {road_curve_radius:.1f} meters"

    cv2.putText(overlay, f"Drift: {abs(center_offset_meters):.2f}m {direction}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, curve_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 7. Save the final evidence
cv2.imwrite('ai_final_output.jpg', overlay)
print("DONE! Check 'ai_final_output.jpg' for the mapped classes, legend, and telemetry.")

# 8. VRAM Cleanup
del model
del processor
del inputs
del outputs
del prediction

gc.collect()
torch.cuda.empty_cache()
print("GPU VRAM has been cleared.")