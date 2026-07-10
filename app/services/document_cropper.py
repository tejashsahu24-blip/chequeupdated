from pathlib import Path
import cv2
import numpy as np
from PIL import Image as PILImage

from ..utils.exceptions import PipelineError


class DocumentCropper:
    @staticmethod
    def _save_image(image, path):
        """Save image supporting both PIL Image and numpy array formats."""
        try:
            if isinstance(image, PILImage.Image):
                image.save(str(path))
            elif isinstance(image, np.ndarray):
                success = cv2.imwrite(str(path), image)
                if not success:
                    raise ValueError(f"cv2.imwrite failed for path: {path}")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save image to {path}: {exc}")

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
            try:
                self._save_image(image[y1:y2, x1:x2], crop_path)
            except Exception as exc:
                crops.append({
                    "label": detection["label"],
                    "confidence": detection["confidence"],
                    "box": detection["box"],
                    "path": None,
                    "error": f"Failed to save crop: {exc}"
                })
                continue
            crops.append({
                "label": detection["label"],
                "confidence": detection["confidence"],
                "box": detection["box"],
                "path": str(crop_path),
                "error": None
            })

        return crops
