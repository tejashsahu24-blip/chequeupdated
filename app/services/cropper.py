import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage


class Cropper:

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

    @staticmethod
    def crop_fields(image_path, detections, output_prefix=None):

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        output_folder = Path(__file__).resolve().parents[1] / "outputs"

        output_folder.mkdir(parents=True, exist_ok=True)

        cropped_images = {}

        for index, detection in enumerate(detections, start=1):

            class_name = detection["class"]

            x1, y1, x2, y2 = detection["box"]

            crop = image[y1:y2, x1:x2]

            file_prefix = f"{output_prefix}_" if output_prefix else ""
            save_path = output_folder / f"{file_prefix}{class_name}_{index}.jpg"

            Cropper._save_image(crop, save_path)

            cropped_images[f"{class_name}_{index}"] = str(save_path)

        return cropped_images
