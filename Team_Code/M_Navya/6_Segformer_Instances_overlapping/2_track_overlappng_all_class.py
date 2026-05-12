from ultralytics import YOLO

# Upgraded from 'n' (nano) to 's' (small) for better stability
model = YOLO('yolo11s-seg.pt')

model.track(
    source='test_vedio.mp4', 
    save=True,                  
    conf=0.15,                   # Lowered threshold to prevent flickering
    iou=0.5,                    
    classes=[1, 2, 3, 5, 7],
    persist=True,
    tracker="bytetrack.yaml"     # Added ByteTrack to remember lost IDs
)

print("Done! Check the 'runs/segment/track' folder for your output video.")