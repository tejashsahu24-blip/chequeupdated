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

        def detect_region(region):
            if region.size == 0:
                return 0.0

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ink_pixels = cv2.countNonZero(thresholded)
            total_pixels = thresholded.size
            return ink_pixels / total_pixels if total_pixels else 0

        regions = [
            image[int(height * SignatureChecker.REGION_TOP):height, int(width * SignatureChecker.REGION_LEFT):width],
            image[int(height * 0.55):height, 0:width],
            image[int(height * 0.65):height, int(width * 0.35):width]
        ]

        ink_ratio = max(detect_region(region) for region in regions)

        if ink_ratio < SignatureChecker.MIN_INK_RATIO:
            return False, "Signature not detected"

        if ink_ratio > SignatureChecker.MAX_INK_RATIO:
            return False, "Signature region unclear or noisy"

        return True, "Signature detected"
