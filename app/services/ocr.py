import os
from pathlib import Path


class OCRService:

    def __init__(self):
        self.backend = None
        self.ocr = None
        self.pytesseract = None
        self.Image = None

        self._init_pytesseract()
        if self.backend is None:
            self._init_paddle()

        if self.backend is None:
            print(
                "Warning: No working OCR backend available. "
                "PDF text extraction fallback will be used when possible."
            )

    def _init_paddle(self):
        try:
            # Force PaddleOCR to use CPU and avoid newer executor/oneDNN execution paths
            os.environ["FLAGS_use_gpu"] = "0"
            os.environ["FLAGS_new_executor"] = "0"
            os.environ["FLAGS_use_new_executor"] = "0"
            os.environ["FLAGS_enable_new_executor"] = "0"
            os.environ["FLAGS_use_pure_cinn"] = "0"
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["FLAGS_enable_mkldnn"] = "0"
            os.environ["FLAGS_use_mkldnn_pass"] = "0"
            os.environ["FLAGS_enable_mkldnn_pass"] = "0"

            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang="en",
                device="cpu",
                enable_mkldnn=False,
                enable_cinn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            self.backend = "paddle"
        except Exception as exc:
            print("PaddleOCR initialization failed:", exc)
            self.backend = None

    def _init_pytesseract(self):
        try:
            import pytesseract
            from PIL import Image
        except Exception as exc:
            print("pytesseract import failed:", exc)
            return

        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            print("Tesseract engine not available:", exc)
            return

        self.pytesseract = pytesseract
        self.Image = Image
        self.backend = "pytesseract"

    # ------------------------------------------------------------------
    # Orientation handling & preprocessing
    #
    # Cheque crops (whether coming from a YOLO detection box or a full
    # page render) are frequently rotated 90/180/270 degrees - e.g. a
    # portrait photo of a physically landscape cheque - and are often
    # small/low-resolution. Neither the pytesseract nor the PaddleOCR
    # path previously corrected for this, so OCR would silently read
    # sideways text and return garbage/empty results even though the
    # cheque itself contained perfectly legible fields. The helpers
    # below upscale/contrast-boost the image and try each orientation,
    # scoring the result against common cheque vocabulary so the best
    # orientation is used instead of whatever the crop happened to be.
    # ------------------------------------------------------------------

    _ORIENTATION_KEYWORDS = (
        "BANK", "PAY", "RUPEES", "ACCOUNT", "IFSC", "CHEQUE", "SIGN",
        "ONLY", "BEARER", "ORDER", "BRANCH", "DATE", "VALID", "MICR"
    )
    _CONFIDENT_KEYWORD_HITS = 3
    _MIN_OCR_WIDTH = 1600

    @classmethod
    def _score_orientation(cls, text):
        if not text:
            return 0
        upper = text.upper()
        return sum(1 for kw in cls._ORIENTATION_KEYWORDS if kw in upper)

    def _preprocess_pil_image(self, pil_image):
        """Upscale small crops and boost contrast so OCR has a fair chance."""
        from PIL import ImageOps

        image = pil_image.convert("L")

        if image.width and image.width < self._MIN_OCR_WIDTH:
            scale = max(1, self._MIN_OCR_WIDTH // image.width)
            image = image.resize((image.width * scale, image.height * scale), self.Image.LANCZOS)

        return ImageOps.autocontrast(image)

    def read_text(self, image_path: str):
        if self.backend == "pytesseract":
            return self._read_text_tesseract(image_path)

        if self.backend == "paddle":
            try:
                return self._read_text_paddle(image_path)
            except Exception as exc:
                print("PaddleOCR failed during read_text:", exc)
                self.backend = None
                self._init_pytesseract()

        if self.backend == "pytesseract":
            return self._read_text_tesseract(image_path)

        print("No OCR backend available for read_text; returning empty OCR result.")
        return [[['', ['', '', 0.0]]]]

    def _read_text_tesseract(self, image_path: str):
        img = self.Image.open(image_path)

        best_text = ""
        best_score = -1

        for angle in (0, 90, 180, 270):
            rotated = img.rotate(angle, expand=True) if angle else img
            processed = self._preprocess_pil_image(rotated)

            try:
                text = self.pytesseract.image_to_string(processed, config="--psm 6")
            except Exception as exc:
                print(f"Tesseract failed at rotation {angle}:", exc)
                continue

            score = self._score_orientation(text)

            # Once an orientation clearly reads like a cheque, stop early
            # instead of burning three more OCR passes on every image.
            if score >= self._CONFIDENT_KEYWORD_HITS:
                return [[[None, [text, 0.0]]]]

            if score > best_score:
                best_score = score
                best_text = text

        return [[[None, [best_text, 0.0]]]]

    def _read_text_paddle(self, image_path: str):
        import numpy as np
        import cv2

        original = self.Image.open(image_path)

        best_result = None
        best_score = -1

        for angle in (0, 90, 180, 270):
            rotated = original.rotate(angle, expand=True) if angle else original
            processed = self._preprocess_pil_image(rotated)
            array = cv2.cvtColor(np.array(processed.convert("RGB")), cv2.COLOR_RGB2BGR)

            if hasattr(self.ocr, "predict"):
                result = self.ocr.predict(array)
            elif hasattr(self.ocr, "ocr"):
                result = self.ocr.ocr(array)
            else:
                result = None

            text, _ = self.extract_text(result)
            score = self._score_orientation(text)

            if score >= self._CONFIDENT_KEYWORD_HITS:
                return result

            if score > best_score:
                best_score = score
                best_result = result

        return best_result

    def extract_text(self, result):
        text = []
        confidence = []

        try:
            if not result:
                return "", 0.0

            # PaddleOCR.predict() returns a list of dicts with rec_texts/rec_scores.
            if isinstance(result, list) and result and isinstance(result[0], dict):
                page = result[0]
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])

                for idx, line_text in enumerate(rec_texts):
                    if not line_text:
                        continue
                    text.append(str(line_text))
                    if idx < len(rec_scores):
                        try:
                            confidence.append(float(rec_scores[idx]))
                        except Exception:
                            confidence.append(0.0)

            # Old PaddleOCR ocr() result or pytesseract wrapper output.
            elif isinstance(result, list) and result and isinstance(result[0], list):
                for line in result[0]:
                    if len(line) > 1 and len(line[1]) >= 2:
                        text.append(str(line[1][0]))
                        try:
                            confidence.append(float(line[1][1]))
                        except Exception:
                            confidence.append(0.0)

            # If OCR backend returns raw text directly.
            elif isinstance(result, str):
                text.append(result)
                confidence.append(0.0)

            # Single-page dict-style output containing raw lines.
            elif isinstance(result, dict):
                rec_texts = result.get("rec_texts") or result.get("lines") or result.get("text")
                if isinstance(rec_texts, list):
                    for item in rec_texts:
                        if isinstance(item, dict) and "text" in item:
                            text.append(str(item["text"]))
                            if "confidence" in item:
                                try:
                                    confidence.append(float(item["confidence"]))
                                except Exception:
                                    confidence.append(0.0)
                        elif isinstance(item, str):
                            text.append(item)
                elif isinstance(rec_texts, str):
                    text.append(rec_texts)

        except Exception as e:
            print("OCR Error:", e)

        avg_conf = sum(confidence) / len(confidence) if confidence else 0
        return " ".join(text).strip(), round(avg_conf, 2)