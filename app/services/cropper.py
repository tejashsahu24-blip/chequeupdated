import cv2
from pathlib import Path


class Cropper:

    @staticmethod
    def crop_fields(image_path, detections, output_prefix=None):

        image = cv2.imread(image_path)

        output_folder = Path(__file__).resolve().parents[1] / "outputs"

        output_folder.mkdir(parents=True, exist_ok=True)

        cropped_images = {}

        for index, detection in enumerate(detections, start=1):

            class_name = detection["class"]

            x1, y1, x2, y2 = detection["box"]

            crop = image[y1:y2, x1:x2]

            file_prefix = f"{output_prefix}_" if output_prefix else ""
            save_path = output_folder / f"{file_prefix}{class_name}_{index}.jpg"

            cv2.imwrite(str(save_path), crop)

            cropped_images[f"{class_name}_{index}"] = str(save_path)

        return cropped_images
