import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

print("Initializing RealSense D435i Color & Depth Pipeline...")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ---> THE FIX: Safely check for a monitor BEFORE OpenCV tries to open a window
display_available = bool(os.environ.get('DISPLAY'))

if not display_available:
    print("\n[INFO] No physical monitor detected via SSH. Running in headless mode.")
    print("[INFO] Images will be continuously saved to 'vision_test.jpg'.\n")

try:
    profile = pipeline.start(config)
    print("Cameras connected! Capturing frames...\n")
    
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        h, w = color_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        distance = depth_frame.get_distance(center_x, center_y)

        cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(color_image, f"{distance:.3f} meters", (center_x + 15, center_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        combined_images = np.hstack((color_image, depth_colormap))

        # ---> THE FIX: Safe conditional display logic
        if display_available:
            cv2.imshow('RealSense Drone Vision', combined_images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.imwrite('vision_test.jpg', combined_images)
            print(f"\rTarget Depth: {distance:.3f} m | Frame saved. Open 'vision_test.jpg' to view! Press Ctrl+C to stop.", end="")
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nVision test terminated by user.")

finally:
    pipeline.stop()
    if display_available:
        cv2.destroyAllWindows()
    print("\nCamera pipeline safely closed.")
