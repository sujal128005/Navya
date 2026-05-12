import cv2
import numpy as np

# ==========================================
# 1. SETUP VIDEO
# ==========================================
video_path = 'test_drive3.mp4'  # Make sure this is in your folder
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

target_h = 1024 # Matches the processing height in your main script
bev_width, bev_height = 400, 400

def nothing(x):
    pass

cv2.namedWindow('Video BEV Tuner', cv2.WINDOW_NORMAL)

# Default starting values for a low-mounted camera
cv2.createTrackbar('TOP H', 'Video BEV Tuner', 47, 100, nothing)          
cv2.createTrackbar('TOP W_OFFSET', 'Video BEV Tuner', 12, 50, nothing)   
cv2.createTrackbar('BOT H', 'Video BEV Tuner', 95, 100, nothing)          
cv2.createTrackbar('BOT W_OFFSET', 'Video BEV Tuner', 2, 50, nothing)    

print("\n=== VIDEO BEV TUNING TOOL ===")
print("[SPACEBAR] : Pause / Play")
print("[ N ]      : Next frame (while paused)")
print("[ Q ]      : Quit and get final code")

paused = True
ret, frame = cap.read()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            # Loop video back to start if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

    # Resize frame to match your Segformer processing resolution
    orig_h, orig_w = frame.shape[:2]
    scale_factor = target_h / orig_h
    final_width = int(orig_w * scale_factor)
    final_height = target_h
    
    resized_frame = cv2.resize(frame, (final_width, final_height))
    h, w = final_height, final_width

    tuner_image = resized_frame.copy()

    # ==========================================
    # 2. READ TRACKBARS & CALCULATE POINTS
    # ==========================================
    top_h_pct = cv2.getTrackbarPos('TOP H', 'Video BEV Tuner') / 100.0
    top_w_off_pct = cv2.getTrackbarPos('TOP W_OFFSET', 'Video BEV Tuner') / 100.0
    bot_h_pct = cv2.getTrackbarPos('BOT H', 'Video BEV Tuner') / 100.0
    bot_w_off_pct = cv2.getTrackbarPos('BOT W_OFFSET', 'Video BEV Tuner') / 100.0

    top_y = h * top_h_pct
    bot_y = h * bot_h_pct
    tl_x = w * (0.5 - top_w_off_pct)
    tr_x = w * (0.5 + top_w_off_pct)
    bl_x = w * (0.5 - bot_w_off_pct)
    br_x = w * (0.5 + bot_w_off_pct)

    src_points = np.float32([
        [tl_x, top_y], [tr_x, top_y],
        [bl_x, bot_y], [br_x, bot_y]
    ])

    dst_points = np.float32([
        [0, 0], [bev_width, 0],
        [0, bev_height], [bev_width, bev_height]
    ])

    # ==========================================
    # 3. DRAW AND WARP
    # ==========================================
    # Draw the green sampling box on the video
    pts = np.int32(src_points).reshape((-1, 1, 2))
    cv2.polylines(tuner_image, [pts], True, (0, 255, 0), 3)

    # Compute BEV
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_output = cv2.warpPerspective(resized_frame, matrix, (bev_width, bev_height))

    # Combine side-by-side
    bev_resized = cv2.resize(bev_output, (int(bev_output.shape[1] * (h/bev_height)), h))
    combined_display = np.hstack((tuner_image, bev_resized))

    # Add Play/Pause status text
    status_text = "PAUSED - Tune Trackbars" if paused else "PLAYING - Press Space to Pause"
    color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(combined_display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Video BEV Tuner', combined_display)

    # ==========================================
    # 4. KEYBOARD CONTROLS
    # ==========================================
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') or key == 32: # Spacebar
        paused = not paused
    elif key == ord('n') and paused:   # Next frame
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

# Output the final parameters
print("\n=== TUNING COMPLETE ===")
print("Copy and paste this block into your MAIN SegFormer script:")
print("-" * 40)
print(f"        src_points = np.float32([")
print(f"            [w * {0.5 - top_w_off_pct:.4f}, h * {top_h_pct:.4f}],")
print(f"            [w * {0.5 + top_w_off_pct:.4f}, h * {top_h_pct:.4f}],")
print(f"            [w * {0.5 - bot_w_off_pct:.4f}, h * {bot_h_pct:.4f}],")
print(f"            [w * {0.5 + bot_w_off_pct:.4f}, h * {bot_h_pct:.4f}]")
print(f"        ])")
print("-" * 40)