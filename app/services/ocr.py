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

    def read_text(self, image_path: str):
        if self.backend == "pytesseract":
            return self._read_text_tesseract(image_path)

        if self.backend == "paddle":
            try:
                if hasattr(self.ocr, "predict"):
                    return self.ocr.predict(image_path)
                if hasattr(self.ocr, "ocr"):
                    return self.ocr.ocr(image_path)
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
        return [[[None, [text, 0.0]]]]

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
