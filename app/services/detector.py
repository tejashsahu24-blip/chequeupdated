from ultralytics import YOLO
from pathlib import Path


class ChequeDetector:

    def __init__(self):
        print("Loading YOLO model...")

        model_path = Path(__file__).resolve().parents[2] / "yolov8n.pt"
        self.model = YOLO(str(model_path))

        print("Model Loaded Successfully")

    def detect(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=0.40,
            save=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = self.model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                })

        return detections
