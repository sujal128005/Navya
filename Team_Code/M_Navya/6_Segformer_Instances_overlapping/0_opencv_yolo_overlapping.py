import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO('yolov8n-seg.pt')

# 2. Open video source (using your specific filename)
video_path = "test_vedio.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open {video_path}")
    exit()

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 3. Output Settings - Using .mp4 with 'mp4v' or .avi with 'XVID'
output_path = 'navya_perception_task.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing: Handling overlaps and stabilizing IDs...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. Optimized Tracking Call
    # tracker="botsort.yaml" is superior for overlapping objects
    # persist=True is CRITICAL for unique ID maintenance
    # iou=0.5 allows the tracker to be more flexible during overlaps
    results = model.track(
        source=frame, 
        persist=True, 
        tracker="botsort.yaml", 
        conf=0.3,  # Lowered slightly to prevent "dropping" IDs during overlap
        iou=0.5, 
        retina_masks=True
    )

    # 5. Visualize results
    # Each instance will have a unique ID and a distinct colored mask
    annotated_frame = results[0].plot()

    # Log IDs to terminal for verification
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        # print(f"Current Tracking IDs: {ids}")

    # 6. Save and Display
    out.write(annotated_frame)
    cv2.imshow("Instance Segmentation & Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Task finished. Video saved to {output_path}")