from pathlib import Path
from ultralytics import YOLO

from ..config import Settings
from ..utils.exceptions import PipelineError


class YOLODetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not self.settings.model_path.exists():
            raise PipelineError(
                "YOLO model file was not found",
                "MODEL_NOT_FOUND",
                {"model_path": str(self.settings.model_path)}
            )
        self.model = YOLO(str(self.settings.model_path))

    def detect(self, image_path: Path) -> list[dict]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.settings.yolo_confidence_threshold,
            save=False,
            verbose=False
        )

        detections: list[dict] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                detections.append({
                    "label": self.model.names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })

        return detections
