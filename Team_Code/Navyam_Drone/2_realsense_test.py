import pyrealsense2 as rs
import numpy as np
import time

print("Initializing RealSense D435i Depth Pipeline...")

# 1. Configure the depth stream
pipeline = rs.pipeline()
config = rs.config()

# Request standard VGA resolution at 30 frames per second
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    # 2. Start streaming
    profile = pipeline.start(config)
    print("Camera successfully connected! Streaming depth data...\n")
    
    # 3. Read frames in a loop
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue

        # Get frame dimensions
        width = depth_frame.get_width()
        height = depth_frame.get_height()

        # Calculate the exact center pixel
        center_x = width // 2
        center_y = height // 2

        # Extract the distance in meters at that specific pixel
        distance = depth_frame.get_distance(center_x, center_y)

        # Print the live output (overwriting the same line in the terminal)
        print(f"\rCenter Pixel Depth: {distance:.3f} meters | Press Ctrl+C to stop", end="")
        time.sleep(0.1)

except RuntimeError as e:
    print(f"\n[ERROR] Hardware not found. Is the USB 3.0 cable plugged in?")
    print(str(e))

except KeyboardInterrupt:
    print("\nVision test terminated by user.")

finally:
    # Safely close the camera connection
    pipeline.stop()
    print("Pipeline closed.")
