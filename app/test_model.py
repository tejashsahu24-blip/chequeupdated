from ultralytics import YOLO

print("Loading model...")

model = YOLO("app/models/cheque_yolov8.pt")

print("✅ Model Loaded Successfully")