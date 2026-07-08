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
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_use_mkldnn_pass", "0")
            os.environ.setdefault("FLAGS_enable_mkldnn", "0")
            os.environ.setdefault("FLAGS_use_gpu", "0")
            os.environ.setdefault("FLAGS_new_executor", "0")
            os.environ.setdefault("FLAGS_use_new_executor", "0")
            os.environ.setdefault("FLAGS_enable_new_executor", "0")
            os.environ.setdefault("FLAGS_use_pure_cinn", "0")

            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang="en",
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

    def read_text(self, image_path: str):
        if self.backend == "pytesseract":
            return self._read_text_tesseract(image_path)

        if self.backend == "paddle":
            try:
                if hasattr(self.ocr, "ocr"):
                    return self.ocr.ocr(image_path)
                if hasattr(self.ocr, "predict"):
                    return self.ocr.predict(image_path)
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
        text = self.pytesseract.image_to_string(img)
        return [[['', ['', text, 0.0]]]]

    def extract_text(self, result):
        text = []
        confidence = []

        try:
            if result and result[0]:
                for line in result[0]:
                    if len(line) > 1 and len(line[1]) >= 2:
                        text.append(line[1][0])
                        try:
                            confidence.append(float(line[1][1]))
                        except Exception:
                            confidence.append(0.0)
        except Exception as e:
            print("OCR Error:", e)

        avg_conf = sum(confidence) / len(confidence) if confidence else 0
        return " ".join(text).strip(), round(avg_conf, 2)
