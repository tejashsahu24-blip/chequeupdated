from pathlib import Path
import cv2

from ..utils.exceptions import PipelineError


class DocumentCropper:
    def crop(self, image_path: Path, detections: list[dict], output_dir: Path) -> list[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            raise PipelineError("Page image could not be read", "INVALID_PAGE_IMAGE", {"image_path": str(image_path)})

        crops: list[dict] = []
        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["box"]
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                crops.append({
                    "label": detection["label"],
                    "confidence": detection["confidence"],
                    "box": detection["box"],
                    "path": None,
                    "error": "Invalid bounding box"
                })
                continue

            label = detection["label"].replace(" ", "_").lower()
            crop_path = output_dir / f"{label}_{index}.jpg"
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            crops.append({
                "label": detection["label"],
                "confidence": detection["confidence"],
                "box": detection["box"],
                "path": str(crop_path),
                "error": None
            })

        return crops
