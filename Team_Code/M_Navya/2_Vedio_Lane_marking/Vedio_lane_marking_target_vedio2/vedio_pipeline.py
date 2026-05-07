import torch
import cv2
import numpy as np
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# 1. INITIALIZATION (Run Once)
# ==========================================
video_path = 'test_drive.mp4'
if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit()

print("Loading SegFormer into RTX 3050 VRAM...")
model_name = "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to("cuda")

# Setup Video Input and Output
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create the Video Writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('autonomous_telemetry_out.mp4', fourcc, fps, (width, height))
out_canny = cv2.VideoWriter('canny_edges_out.mp4', fourcc, fps, (width, height))

# --- NEW: ROI VIDEO WRITER ---
out_roi = cv2.VideoWriter('roi_vision_out.mp4', fourcc, fps, (width, height))

print("Processing Video... Press 'q' to stop.")

# ==========================================
# 2. THE PROCESSING LOOP (Runs every frame)
# ==========================================
with torch.no_grad(): # Keep gradient calculation off for speed

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # --- CANNY EDGE PROCESSING ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out_canny.write(edges_bgr)
        
        # --- A. AI SEGMENTATION ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to("cuda")
        
        outputs = model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.logits, size=(height, width), mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        road_mask = np.where(prediction == 0, 255, 0).astype(np.uint8)
        
        # --- B. EDGE DETECTION & ROI (OUTLIER PROOF) ---
        roi_top = int(height * 0.55) 
        roi_bottom = int(height * 0.95)

        left_x, left_y, right_x, right_y = [], [], [], []

        # Scan the AI mask ONLY within the safe zone
        for y in range(roi_top, roi_bottom):
            row_pixels = np.where(road_mask[y, :] == 255)[0]

            if len(row_pixels) > 20:
                split_indices = np.where(np.diff(row_pixels) > 1)[0] + 1
                chunks = np.split(row_pixels, split_indices)
                largest_chunk = max(chunks, key=len)
                
                if len(largest_chunk) > 50:
                    left_x.append(largest_chunk[0])
                    left_y.append(y)
                    right_x.append(largest_chunk[-1])
                    right_y.append(y)

        # Convert to numpy arrays
        left_x = np.array(left_x)
        left_y = np.array(left_y)
        right_x = np.array(right_x)
        right_y = np.array(right_y)

        # Prepare the primary overlay (Clean drivable area mask)
        overlay = frame.copy()
        overlay[road_mask == 255] = [0, 200, 0] 
        
        # --- NEW: ROI VISUALIZATION GENERATION ---
        # 1. Darken the entire frame
        roi_visual = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
        
        # 2. Restore full brightness to just the ROI section
        roi_visual[roi_top:roi_bottom, :] = frame[roi_top:roi_bottom, :]
        
        # 3. Create a green mask strictly bounded by the ROI
        roi_mask_color = np.zeros_like(frame)
        roi_mask_color[roi_top:roi_bottom, :][road_mask[roi_top:roi_bottom, :] == 255] = [0, 255, 0]
        
        # 4. Blend the mask and draw boundary lines
        roi_visual = cv2.addWeighted(roi_visual, 1.0, roi_mask_color, 0.4, 0)
        cv2.line(roi_visual, (0, roi_top), (width, roi_top), (0, 0, 255), 2)
        cv2.line(roi_visual, (0, roi_bottom), (width, roi_bottom), (0, 0, 255), 2)
        cv2.putText(roi_visual, "ACTIVE SCAN ZONE", (30, roi_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out_roi.write(roi_visual)
        # -----------------------------------------

        # --- C. MATH & TELEMETRY ---
        try:
            if len(left_x) > 50 and len(right_x) > 50:
                
                left_fit = np.polyfit(left_y, left_x, 2)
                right_fit = np.polyfit(right_y, right_x, 2)
            
                # Real-World Telemetry (Center Offset & Curve)
                y_eval = height
                left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
                right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
                
                lane_center = (left_x_bottom + right_x_bottom) / 2
                vehicle_center = width / 2
                offset_pixels = vehicle_center - lane_center
                
                meters_per_pixel = 3.7 / (right_x_bottom - left_x_bottom)
                center_offset_meters = offset_pixels * meters_per_pixel
                
                ym_per_pix = 30.0 / height
                left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * meters_per_pixel, 2)
                right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * meters_per_pixel, 2)
                y_eval_real = height * ym_per_pix
                
                left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_real + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
                right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_real + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
                road_curve_radius = (left_curverad + right_curverad) / 2

                # --- D. THE HUD (Heads Up Display) ---
                direction = "Right" if center_offset_meters > 0 else "Left"
                offset_text = f"Drift: {abs(center_offset_meters):.2f}m {direction}"
                curve_text = f"Radius: {road_curve_radius:.1f}m" if road_curve_radius < 3000 else "Radius: Straight"
                
                cv2.putText(overlay, offset_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(overlay, curve_text, (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        except Exception as e:
            print(f"Dropped frame math: {e}")

        # Blend and display main telemetry
        final_frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        out.write(final_frame)
        
        # Display screens dynamically
        display_frame = cv2.resize(final_frame, (854, 480)) 
        display_canny = cv2.resize(edges, (854, 480))
        display_roi = cv2.resize(roi_visual, (854, 480)) # --- NEW: DISPLAY ROI LIVE ---
        
        cv2.imshow('Autonomous Telemetry (Clean)', display_frame)
        cv2.imshow('Canny Edge Vision', display_canny)
        cv2.imshow('ROI Scanner', display_roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ==========================================
# 3. CLEANUP
# ==========================================
cap.release()
out.release()
out_canny.release()
out_roi.release() # --- NEW: RELEASE ROI WRITER ---
cv2.destroyAllWindows()
print("Pipeline Offline.")
print("Outputs saved: 'autonomous_telemetry_out.mp4', 'canny_edges_out.mp4', and 'roi_vision_out.mp4'")