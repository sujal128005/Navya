import torch
import cv2
import os
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. Path Check
img_path = 'test_road.jpg'
if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
    exit()

# 2. Load Model
model_name = "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to("cuda")

# 3. Process Image
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
inputs = processor(images=image_rgb, return_tensors="pt").to("cuda")

print("processing...")
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.nn.functional.interpolate(
        outputs.logits, size=image.shape[:2], mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].cpu().numpy()

# 4. Create the Mask
# Label 0 is Road, Label 13 is Car
road_mask = np.where(prediction == 0, 255, 0).astype(np.uint8)
car_mask = np.where(prediction == 13, 255, 0).astype(np.uint8)

# 1. Get the edges of the AI mask
# This creates a 1-pixel wide line following the road boundary
# 1. Get the edges of the AI mask
edges = cv2.Canny(road_mask, 100, 200)

# Erase the bottom border noise (You did this perfectly last time!)
edges[-20:, :] = 0 

# 2. Divide and Conquer
height, width = edges.shape
midpoint = width // 2
left_lane_pixels = edges[:, :midpoint]
right_lane_pixels = edges[:, midpoint:]

# --- THE HORIZON MAGNET FIX ---
# Your horizon is very low! We must cut off the top 60% to destroy the top border.
roi_top = int(height * 0.6) 

# 3. Extract Coordinates ONLY from the safe bottom zone
left_y, left_x = np.nonzero(left_lane_pixels[roi_top:, :])
right_y, right_x = np.nonzero(right_lane_pixels[roi_top:, :])

# 4. Adjust the Y-coordinates back to their real positions
left_y = left_y + roi_top
right_y = right_y + roi_top
right_x = right_x + midpoint

    

# 5. Polynomial Fitting
if len(left_x) > 50:
    left_fit = np.polyfit(left_y, left_x, 2)
    
if len(right_x) > 50:
    right_fit = np.polyfit(right_y, right_x, 2)
# --- FIXED SECTION: CREATE OVERLAY FIRST ---
# --- NEW: CALCULATE CENTER OFFSET ---

# 1. Define the bottom of the image (y = height)
y_eval = height

# 2. Calculate where the left and right lines are at the very bottom of the screen
# Using your formula: x = Ay² + By + C
left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

# 3. Find the true center of the lane
lane_center = (left_x_bottom + right_x_bottom) / 2

# 4. Find the center of the vehicle (middle of the image)
vehicle_center = width / 2

# 5. Calculate the difference in pixels
offset_pixels = vehicle_center - lane_center

# 6. Convert pixels to meters (Estimation: standard lane is ~3.7 meters wide)
# We calculate how many pixels wide the lane is at the bottom, then divide 3.7 by that
lane_width_pixels = right_x_bottom - left_x_bottom
meters_per_pixel = 3.7 / lane_width_pixels

# Final Real-World Offset!
center_offset_meters = offset_pixels * meters_per_pixel

# Print the result so you can see it in the terminal
direction = "Right" if center_offset_meters > 0 else "Left"
print(f"Vehicle is {abs(center_offset_meters):.2f} meters off center to the {direction}.")

# --- END OF OFFSET CALCULATION ---
# --- NEW: CALCULATE CURVE RADIUS ---

# 1. Real-World Conversion Factors
# We know xm_per_pix from your previous step. 
# We will estimate that the height of the image shows about 30 meters of road ahead.
xm_per_pix = meters_per_pixel 
ym_per_pix = 30.0 / height

# 2. Re-fit the polynomials in REAL METERS instead of pixels
# We multiply the raw coordinates by our conversion factors to scale the math into the real world.
left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)

# 3. Define where to measure the radius
# We want the radius right at the bottom of the screen where the drone currently is.
y_eval_real = height * ym_per_pix

# 4. Apply the Calculus Formula: R = [1 + (2Ay + B)^2]^(3/2) / |2A|
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_real + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_real + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

# 5. Average the two lanes to find the true curve of the road
road_curve_radius = (left_curverad + right_curverad) / 2

# Print the final telemetry data
if road_curve_radius > 3000:
    print("The road is essentially STRAIGHT.")
else:
    print(f"Road Curve Radius: {road_curve_radius:.1f} meters")

# --- END OF CURVE RADIUS CALCULATION ---

# 1. Create the base overlay (Copy the original image)
overlay = image.copy()

# 2. Add the AI Colors (Green for road, Red for cars)
overlay[road_mask == 255] = [0, 255, 0] 
overlay[car_mask == 255] = [0, 0, 255]

# --- FIXED DRAWING BLOCK ---

# 1. Start drawing from the horizon (roi_top) down to the bottom (height)
# We no longer start at 0 (the top of the image)
plot_y = np.linspace(roi_top, height-1, int(height - roi_top))

# 2. Calculate x-values using your math: x = Ay² + By + C
left_fitx = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
right_fitx = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

# 3. Format the points so OpenCV can understand them
left_pts = np.array([np.transpose(np.vstack([left_fitx, plot_y]))], np.int32)
right_pts = np.array([np.transpose(np.vstack([right_fitx, plot_y]))], np.int32)

# 4. Draw the lines directly onto your 'overlay' image
cv2.polylines(overlay, left_pts, isClosed=False, color=(0, 255, 255), thickness=10)
cv2.polylines(overlay, right_pts, isClosed=False, color=(255, 255, 255), thickness=10)

# --- END OF DRAWING BLOCK ---
# 7. Save the final evidence
cv2.imwrite('final_output.jpg', overlay)
cv2.imwrite('road_mask.jpg', road_mask)

print("DONE! Check 'final_output.jpg' for the road mask AND lane lines.")