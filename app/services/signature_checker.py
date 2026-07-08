import cv2


class SignatureChecker:
    """
    Heuristic signature presence check.

    A signature detection model isn't part of this project yet, so this
    checker looks at the bottom-right area of the cheque image (where a
    signature is normally placed) and measures how much dark "ink" is
    present there. A blank region means no signature; too much ink means
    the crop is noisy/unclear rather than a clean signature.
    """

    # Region of the cheque (as a fraction of width/height) where a
    # signature is typically found.
    REGION_TOP = 0.65
    REGION_LEFT = 0.55

    MIN_INK_RATIO = 0.01
    MAX_INK_RATIO = 0.55

    @staticmethod
    def check_signature(image):

        if image is None or image.size == 0:
            return False, "Image not available for signature check"

        height, width = image.shape[:2]

        y1 = int(height * SignatureChecker.REGION_TOP)
        x1 = int(width * SignatureChecker.REGION_LEFT)

        region = image[y1:height, x1:width]

        if region.size == 0:
            return False, "Signature region not found"

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        ink_pixels = cv2.countNonZero(thresholded)
        total_pixels = thresholded.size

        ink_ratio = ink_pixels / total_pixels if total_pixels else 0

        if ink_ratio < SignatureChecker.MIN_INK_RATIO:
            return False, "Signature not detected"

        if ink_ratio > SignatureChecker.MAX_INK_RATIO:
            return False, "Signature region unclear or noisy"

        return True, "Signature detected"
