import os
import re
from pathlib import Path
from PIL import Image, ImageOps


class OCRService:

    def __init__(self):
        self.backend = None
        self.ocr = None
        self.pytesseract = None

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
            # PaddleX defaults to a cache under the user's home directory.
            # That location is commonly read-only for services/IDE launches,
            # which makes PaddleOCR fail during model initialisation and leaves
            # the API with no OCR backend.  Keep the cache with the app so the
            # process that runs the API owns both the models and their locks.
            paddle_cache = Path(__file__).resolve().parents[1] / "models" / "paddle"
            paddle_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paddle_cache))
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

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
        except Exception as exc:
            print("pytesseract import failed:", exc)
            return

        try:
            # The Windows installer does not always add Tesseract to the PATH
            # of an already-running VS Code process.  Resolve the standard
            # installation location explicitly before probing the executable.
            configured_command = os.getenv("TESSERACT_CMD")
            standard_command = Path(os.getenv("ProgramFiles", r"C:\\Program Files")) / "Tesseract-OCR" / "tesseract.exe"
            if configured_command:
                pytesseract.pytesseract.tesseract_cmd = configured_command
            elif standard_command.exists():
                pytesseract.pytesseract.tesseract_cmd = str(standard_command)

            pytesseract.get_tesseract_version()
        except Exception as exc:
            print("Tesseract engine not available:", exc)
            return

        self.pytesseract = pytesseract
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
        keyword_score = sum(1 for kw in cls._ORIENTATION_KEYWORDS if kw in upper)
        # A field-shaped value is stronger evidence than a generic cheque
        # word.  This also makes sparse-text OCR (PSM 11) selectable for
        # cheques whose printed template contains little readable prose.
        field_score = sum((
            bool(re.search(r"[A-Z]{4}\s*[0O]\s*(?:[A-Z0-9]\s*){6}", upper)),
            bool(re.search(r"\b\d{9,18}\b", upper)),
            bool(re.search(r"\b\d{6}\b", upper)),
            bool(re.search(r"\b\d{9}\b", upper)),
        ))
        return keyword_score + (field_score * 3)

    def _preprocess_pil_image(self, pil_image):
        """Upscale small crops and boost contrast so OCR has a fair chance."""
        image = pil_image.convert("L")

        if image.width and image.width < self._MIN_OCR_WIDTH:
            scale = max(1, self._MIN_OCR_WIDTH // image.width)
            image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)

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
        img = Image.open(image_path)

        best_text = ""
        best_score = -1

        for angle in (0, 90, 180, 270):
            rotated = img.rotate(angle, expand=True) if angle else img
            processed = self._preprocess_pil_image(rotated)

            # PSM 6 handles a clean block of text.  PSM 11/12 handle the
            # scattered labels and MICR line found on most cheque templates.
            for psm in (6, 11, 12):
                try:
                    text = self.pytesseract.image_to_string(processed, config=f"--psm {psm}")
                except Exception as exc:
                    print(f"Tesseract failed at rotation {angle}, PSM {psm}:", exc)
                    continue

                score = self._score_orientation(text)
                if score > best_score:
                    best_score = score
                    best_text = text

                # A field-shaped read plus cheque context is a reliable
                # result; no further rotations are needed.
                if score >= self._CONFIDENT_KEYWORD_HITS + 3:
                    return [[[None, [text, 0.0]]]]

        return [[[None, [best_text, 0.0]]]]

    def _read_text_paddle(self, image_path: str):
        import numpy as np
        import cv2

        original = Image.open(image_path)

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

    def read_cheque_field_hints(self, image_path: str) -> str:
        """Read cheque regions that full-page OCR commonly misses.

        Indian cheque serial numbers are printed in a MICR-like font along
        the lower-left edge, while the CTS-2010 mark is often vertical on the
        left edge.  Reading those small areas with their own OCR settings is
        much more reliable than asking a page-layout OCR pass to find them.
        """
        if self.backend != "pytesseract":
            return ""

        try:
            image = Image.open(image_path).convert("L")
            width, height = image.size
            if width < 100 or height < 100:
                return ""

            # Serial number is normally in the left portion of the MICR line.
            serial_region = image.crop((
                int(width * 0.14), int(height * 0.76),
                int(width * 0.48), int(height * 1.00),
            ))
            serial_region = self._preprocess_pil_image(serial_region)
            # serial_region.save("serial_region.png")  # Save the serial region for debugging
            serial_text = self.pytesseract.image_to_string(
                serial_region,
                config="--psm 7 -c tessedit_char_whitelist=0123456789OQDILZSBG",
            )

            # CTS-2010 is commonly printed vertically on the left border.
            cts_region = image.crop((
                0, int(height * 0.28), int(width * 0.18), int(height * 0.66)
            ))
            cts_texts = []
            for angle in (90, 270):
                rotated = self._preprocess_pil_image(cts_region.rotate(angle, expand=True))
                cts_texts.append(self.pytesseract.image_to_string(rotated, config="--psm 6"))

            hints = []
            # On the MICR line the cheque serial is the first six digits. OCR
            # commonly reads the adjacent MICR separator as letters, so
            # normalise those symbols before taking that serial component.
            serial_digits = serial_text.upper().translate(str.maketrans({
                "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                "Z": "2", "S": "5", "B": "8", "G": "6",
            }))
            serial_digits = re.sub(r"[^0-9]", "", serial_digits)
            if len(serial_digits) >= 6:
                hints.append(f"Cheque No: {serial_digits[:6]}")

            cts_text = " ".join(cts_texts).upper()
            # Rotated OCR can read CTS as SLO/SIO and 2010 backwards as 0102.
            # Those signatures are specific to the printed vertical CTS mark.
            if (re.search(r"(?:CTS|SLO|SIO)", cts_text)
                    and re.search(r"(?:20[0-9]{2}|[0-9]{2}02|010[0-9])", cts_text)):
                hints.append("CTS-2010")

            # Full-page OCR can mistake MICR digits for an account number.
            # Prefer the printed A/c box in the middle/lower-left area.  The
            # region is deliberately broad enough for common Indian cheque
            # templates while excluding the MICR line at the very bottom.
            account_region = image.crop((
                int(width * 0.03), int(height * 0.45),
                int(width * 0.45), int(height * 0.65),
            ))
            account_texts = []
            for psm in (6, 11):
                account_texts.append(self.pytesseract.image_to_string(
                    self._preprocess_pil_image(account_region), config=f"--psm {psm}"
                ))

            for account_text in account_texts:
                # Accept OCR's usual look-alike characters only in this
                # numeric field, then retain a 9-18 digit account candidate.
                normalized = account_text.upper().translate(str.maketrans({
                    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                    "Z": "2", "S": "5", "B": "8", "G": "6",
                }))
                candidates = re.findall(r"(?<!\d)\d[\d\s-]{7,20}\d(?!\d)", normalized)
                if candidates:
                    digits = re.sub(r"[^0-9]", "", candidates[0])
                    if 9 <= len(digits) <= 18:
                        hints.append(f"Account No: {digits}")
                        break

            # The payee line is a better customer-name source than bank
            # boilerplate or an often illegible signature at the bottom.
            payee_region = image.crop((
                int(width * 0.75), int(height * 0.50),
                int(width), int(height),
            ))
            payee_region.save("payee_region.png")

            payee_text = self.pytesseract.image_to_string(
                self._preprocess_pil_image(payee_region), config="--psm 6"
            )
            print("Payee OCR result:", payee_text)  # Debugging output
            if payee_text.strip():
                hints.append(f"Customer Name: {payee_text.strip()}")
            return " ".join(hints)
        except Exception as exc:
            print("Cheque field OCR failed:", exc)
            return ""

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
