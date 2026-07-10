from pathlib import Path
from ultralytics import YOLO

from ..config import get_settings
from ..utils.exceptions import PipelineError


class ChequeDetector:

    def __init__(self):
        print("Loading YOLO model...")

        settings = get_settings()
        model_path = settings.model_path
        if not model_path.exists():
            raise PipelineError(
                "YOLO model file was not found",
                "MODEL_NOT_FOUND",
                {"model_path": str(model_path)}
            )

        self.model = YOLO(str(model_path))
        self.confidence_threshold = settings.yolo_confidence_threshold
        class_names = self.model.names.values() if isinstance(self.model.names, dict) else self.model.names
        self.supports_cheque_detection = any(
            str(name).strip().lower() == "cheque" for name in class_names
        )

        print(f"Model Loaded Successfully: {model_path}")
        if not self.supports_cheque_detection:
            print("YOLO model has no 'cheque' class; using full-page OCR fallback.")

    def detect(self, image_path):
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            save=False,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id].lower()

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
