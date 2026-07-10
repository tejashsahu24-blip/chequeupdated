import cv2
import os


class ImageQuality:

    @staticmethod
    def check_resolution(image):

        height, width = image.shape[:2]

        if width < 1000 or height < 500:
            return False, "Low Resolution"

        return True, "Resolution OK"


    @staticmethod
    def check_file_size(image_path):

        size = os.path.getsize(image_path)

        size_mb = size / (1024 * 1024)

        if size_mb > 25:
            return False, "Image Size Too Large"

        # Small scanned images are still processable: the OCR path upscales
        # them and the quality checks decide whether values can be trusted.
        # Reject only an actually empty upload, rather than discarding a
        # useful cheque before extraction is attempted.
        if size <= 0:
            return False, "Uploaded file is empty"

        return True, "File Size OK"


    @staticmethod
    def check_blur(image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if score < 100:
            return False, "Blur Image"

        return True, "Image Clear"


    @staticmethod
    def check_brightness(image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()

        if brightness < 50:
            return False, "Image Too Dark"

        if brightness > 220:
            return False, "Image Too Bright"

        return True, "Brightness OK"
