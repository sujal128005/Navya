from ultralytics import YOLO

# 1. Load the Instance Segmentation model
model = YOLO('yolo11n-seg.pt')

# 2. Process the video automatically
# YOLO will read, annotate, and save the video without needing a manual loop
model.track(
    source='test_vedio.mp4', # Your input video file
    save=True,                  # Tells YOLO to save the annotated video
    conf=0.25,                  # Filters out low-confidence guesses
    iou=0.5,                    # Ensures overlapping cars get separate boxes
    classes=[1,2,3,5, 7]              # 2 = car, 7 = truck
)

print("Done! Check the 'runs/segment/predict' folder for your output video.")