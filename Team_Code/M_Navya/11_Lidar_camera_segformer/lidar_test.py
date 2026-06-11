import pyrealsense2 as rs
import serial
import math
import cv2
import numpy as np

# --- Configuration ---
LIDAR_PORT = 'COM5'
BAUD_RATE = 230400

# --- 1. Initialize RealSense Camera ---
pipeline = rs.pipeline()
config = rs.config()
# Request a standard 640x480 RGB stream at 30 FPS
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# --- 2. Pull Factory Intrinsics ---
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# Build the exact K Matrix from the camera's firmware
K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])

print("\n--- Factory Camera Matrix (K) ---")
print(K)
print("---------------------------------\n")

# --- 3. Physical Extrinsic Matrix [R | t] ---
# You still need to manually measure the physical distance between the camera and LiDAR.
# This assumes the camera is mounted exactly 100mm above the LiDAR, looking forward.
R = np.eye(3) 
t = np.array([[0], [-100], [0]]) # Modify these values based on your physical rig
Extrinsic = np.hstack((R, t))

# --- Setup LiDAR Connection ---
ser = serial.Serial(LIDAR_PORT, BAUD_RATE, timeout=1)

angles_buffer = []
distances_buffer = []

print("Starting Sensor Fusion... Press 'q' in the video window to quit.")

def get_color_by_distance(dist_mm, max_dist=5000):
    norm = min(dist_mm / max_dist, 1.0)
    b = int(255 * norm)
    r = int(255 * (1 - norm))
    return (b, 0, r)

try:
    while True:
        # Read LiDAR Packet
        if ser.read(1) == b'\x54' and ser.read(1) == b'\x2c':
            packet = b'\x54\x2c' + ser.read(45)
            if len(packet) == 47:
                start_angle = (packet[5] << 8 | packet[4]) / 100.0
                end_angle = (packet[43] << 8 | packet[42]) / 100.0
                step = (end_angle - start_angle)
                if step < 0: step += 360.0
                step /= 11.0
                
                for i in range(12):
                    idx = 6 + i * 3
                    distance = packet[idx+1] << 8 | packet[idx]
                    angle = start_angle + step * i
                    if angle >= 360.0: angle -= 360.0
                    
                    if distance > 0:
                        angles_buffer.append(math.radians(angle))
                        distances_buffer.append(distance)
                
                # Once we have a full LiDAR sweep, process a RealSense frame
                if len(angles_buffer) > 300:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    # Convert RealSense frame to OpenCV numpy array
                    frame = np.asanyarray(color_frame.get_data())

                    # Process each collected LiDAR point
                    for i in range(len(angles_buffer)):
                        r = distances_buffer[i]
                        theta = angles_buffer[i]
                        
                        x_l = r * math.cos(theta)
                        y_l = r * math.sin(theta)
                        z_l = 0
                        
                        P_L = np.array([[x_l], [y_l], [z_l], [1.0]])
                        
                        P_C = Extrinsic @ P_L
                        
                        X_cam = -P_C[1, 0] 
                        Y_cam = -P_C[2, 0] 
                        Z_cam = P_C[0, 0]  
                        
                        if Z_cam <= 0:
                            continue
                            
                        P_cam_3d = np.array([[X_cam], [Y_cam], [Z_cam]])
                        p_img = K @ P_cam_3d
                        
                        u = int(p_img[0, 0] / p_img[2, 0])
                        v = int(p_img[1, 0] / p_img[2, 0])
                        
                        height, width = frame.shape[:2]
                        if 0 <= u < width and 0 <= v < height:
                            color = get_color_by_distance(r)
                            cv2.circle(frame, (u, v), 3, color, -1)
                    
                    cv2.imshow("LiDAR + RealSense Fusion", frame)
                    
                    angles_buffer.clear()
                    distances_buffer.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    ser.close()
    pipeline.stop()
    cv2.destroyAllWindows()