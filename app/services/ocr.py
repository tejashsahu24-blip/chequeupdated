import json
import os
import re
import subprocess
from collections import Counter
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

    def _orient_image_for_cheque_fields(self, image):
        """Return the rotation that is most likely to be an upright cheque.

        ``read_text`` already checks all four rotations.  The targeted field
        reader must do the same before using fixed cheque coordinates;
        otherwise an upright crop from a portrait PDF is actually a vertical
        slice through unrelated text.
        """
        best_image = image
        best_score = -1

        for angle in (0, 90, 180, 270):
            rotated = image.rotate(angle, expand=True) if angle else image
            try:
                text = self.pytesseract.image_to_string(
                    self._preprocess_pil_image(rotated), config="--psm 11"
                )
            except Exception:
                continue

            score = self._score_orientation(text)
            if score > best_score:
                best_score = score
                best_image = rotated

        return best_image

    @staticmethod
    def _numeric_sequences(text, min_len=5, max_len=8):
        candidates = []
        seen = set()

        for length in range(max_len, min_len - 1, -1):
            pattern = rf"(?<!\d)(\d(?:[\s-]*\d){{{length - 1}}})(?!\d)"
            for match in re.finditer(pattern, text):
                candidate = re.sub(r"[^0-9]", "", match.group(1))
                if len(candidate) != length or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)

        return candidates

    @staticmethod
    def _first_micr_serial(text):
        """Return the left-most six MICR digits from a targeted OCR read.

        Tesseract commonly reads the MICR font's ``6`` as ``B`` and its
        ``1`` as ``L``.  This conversion is deliberately limited to the
        lower-left MICR crop; applying it to full-page/account OCR would make
        ordinary text and account numbers less reliable.
        """
        normalized = (text or "").upper().translate(str.maketrans({
            "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "S": "5", "B": "6", "G": "6",
        }))
        digits = re.sub(r"[^0-9]", "", normalized)
        return digits[:6] if len(digits) >= 6 else ""

    @staticmethod
    def _select_micr_serial(candidates):
        """Select the most consistently read six-digit MICR serial."""
        valid = [candidate for candidate in candidates if re.fullmatch(r"\d{6}", candidate or "")]
        if not valid:
            return ""

        counts = Counter(valid)
        selected = max(counts, key=lambda candidate: counts[candidate])
        # One noisy OCR result must not become an API value.  At least two
        # independent preprocessing/segmentation passes must agree.
        return selected if counts[selected] >= 2 else ""

    @staticmethod
    def _sdk_micr_serial_from_text(text):
        """Return the first six digits from an SDK-recognized MICR line."""
        digits = re.sub(r"[^0-9]", "", text or "")
        return digits[:6] if len(digits) >= 6 else ""

    @classmethod
    def _sdk_micr_serial_from_payload(cls, payload):
        zones = payload.get("zones") if isinstance(payload, dict) else None
        if not isinstance(zones, list):
            return ""

        zone_texts = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            zone_text = zone.get("text")
            if zone_text:
                zone_texts.append(str(zone_text))

        # The recognizer normally returns the complete MICR row as one zone.
        # If several zones are returned, joining them keeps the left-to-right
        # first six digits while still ignoring routing/account later blocks.
        return cls._sdk_micr_serial_from_text(" ".join(zone_texts))

    def _read_ultimate_micr_serial(self, image_path):
        recognizer_from_env = os.getenv("ULTIMATE_MICR_RECOGNIZER")
        sdk_root = Path(__file__).resolve().parents[2] / "third_party" / "ultimateMICR-SDK"
        recognizer = (
            Path(recognizer_from_env)
            if recognizer_from_env
            else sdk_root / "binaries" / "windows" / "x86_64" / "recognizer.exe"
        )
        assets = sdk_root / "assets"

        if not recognizer.exists() or not assets.exists():
            return ""

        try:
            completed = subprocess.run(
                [
                    str(recognizer),
                    "--image", str(image_path),
                    "--format", "e13b",
                    "--assets", str(assets),
                ],
                cwd=str(recognizer.parent),
                input="`n",
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except Exception as exc:
            print("ultimateMICR recognizer failed:", exc)
            return ""

        output = f"{completed.stdout}\n{completed.stderr}"
        for line in output.splitlines():
            line = line.strip()
            if "result:" not in line:
                continue
            try:
                payload = json.loads(line.split("result:", 1)[1].strip())
            except json.JSONDecodeError:
                continue
            serial = self._sdk_micr_serial_from_payload(payload)
            if serial:
                return serial

        if completed.returncode != 0:
            print("ultimateMICR recognizer exited with code:", completed.returncode)
        return ""
    def _micr_preprocess_variants(self, region):
        """Create threshold variants that preserve thin E-13B MICR strokes."""
        base = self._preprocess_pil_image(region)
        return (
            base.point(lambda pixel: 0 if pixel < 185 else 255),
            base,
            base.point(lambda pixel: 0 if pixel < 145 else 255),
        )

    @staticmethod
    def _detected_micr_regions(image):
        """Locate likely compact MICR rows in the lower cheque area.

        PDF renderers and phone photos place the MICR row at different
        vertical positions.  This uses the actual ink bands instead of a
        bank-template coordinate and returns several bottom candidates so a
        signature line cannot hide the MICR row.
        """
        width, height = image.size
        start_y = int(height * 0.58)
        end_y = int(height * 0.98)
        left = int(width * 0.02)
        # The cheque serial is the left-most MICR block.  Restricting the
        # detected row to this side prevents the account/routing blocks from
        # changing the character segmentation of that six-digit value.
        right = int(width * 0.50)
        region = image.crop((left, start_y, right, end_y))

        pixels = region.load()
        row_counts = [
            sum(1 for x in range(region.width) if pixels[x, y] < 125)
            for y in range(region.height)
        ]
        active_rows = [count > max(12, int(region.width * 0.012)) for count in row_counts]
        groups = []
        group_start = None
        gap = 0
        for index, active in enumerate(active_rows + [False]):
            if active and group_start is None:
                group_start = index
                gap = 0
            elif group_start is not None and not active:
                gap += 1
                # Small blank gaps occur inside a single printed text row.
                if gap > 8:
                    end = index - gap + 1
                    if 12 <= end - group_start <= 140:
                        groups.append((group_start, end))
                    group_start = None
                    gap = 0

        if not groups:
            return []

        crops = []
        # MICR is normally among the final compact rows.  Trying the last
        # three works for both a blank-footer PDF and a tightly cropped photo.
        for top, bottom in reversed(groups[-3:]):
            # Keep generous vertical context around the glyphs.  Tight crops
            # clip the distinctive upper/lower strokes that separate MICR 3,
            # 5, and 7.
            padding = max(12, int((bottom - top) * 1.0))
            crops.append(image.crop((
                left,
                max(0, start_y + top - padding),
                right,
                min(height, start_y + bottom + padding),
            )))
        return crops

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

            image = self._orient_image_for_cheque_fields(image)
            width, height = image.size

            # The MICR serial is printed along the bottom edge.  Read a few
            # overlapping bottom strips so a narrow crop or a slightly skewed
            # page does not hide the cheque number.
            detected_serial_regions = self._detected_micr_regions(image)
            # Scale-relative fallback bands cover documents where low
            # contrast prevents ink-band detection.  They do not assume any
            # one bank's cheque layout.
            fallback_serial_regions = [
                image.crop((
                    int(width * 0.02), int(height * 0.62),
                    int(width * 0.70), int(height * 0.76),
                )),
                image.crop((
                    int(width * 0.02), int(height * 0.72),
                    int(width * 0.70), int(height * 0.86),
                )),
                image.crop((
                    int(width * 0.02), int(height * 0.82),
                    int(width * 0.70), int(height * 0.98),
                )),
            ]

            def read_serial_candidates(serial_region):
                candidates = []
                for processed_region in self._micr_preprocess_variants(serial_region):
                    for config in (
                        "--psm 7 -c tessedit_char_whitelist=0123456789OQDILZSBG",
                        "--psm 6 -c tessedit_char_whitelist=0123456789OQDILZSBG",
                    ):
                        serial_text = self.pytesseract.image_to_string(
                            processed_region,
                            config=config,
                        )
                        # A cheque serial is always the *first six-digit block*
                        # in the MICR line.  Do not retain 5- or 7/8-digit
                        # candidates: those can be an OCR fragment of an account
                        # number or a MICR transaction/routing component.
                        candidates.extend(self._numeric_sequences(serial_text, 6, 6))
                        serial = self._first_micr_serial(serial_text)
                        if serial:
                            candidates.append(serial)
                return candidates

            # CTS-2010 is commonly printed vertically on the left border.
            cts_region = image.crop((
                0, int(height * 0.28), int(width * 0.18), int(height * 0.66)
            ))
            cts_texts = []
            for angle in (90, 270):
                rotated = self._preprocess_pil_image(cts_region.rotate(angle, expand=True))
                cts_texts.append(self.pytesseract.image_to_string(rotated, config="--psm 6"))

            hints = []
            sdk_serial = self._read_ultimate_micr_serial(image_path)
            if sdk_serial:
                hints.append(f"MICR Cheque No: {sdk_serial}")

            # On the MICR line the cheque serial is the first six digits. OCR
            # commonly reads the adjacent MICR separator as letters, so
            # normalise those symbols before taking that serial component.
            # Stop as soon as a compact, dynamically detected MICR row has a
            # verified consensus.  Broad fallback bands are used only when
            # necessary, so unrelated lower-page text cannot outvote it.
            serial = ""
            if not sdk_serial:
                for serial_region in detected_serial_regions + fallback_serial_regions:
                    serial = self._select_micr_serial(read_serial_candidates(serial_region))
                    if serial:
                        break
                if serial:
                    hints.append(f"MICR Cheque No: {serial}")

            cts_text = " ".join(cts_texts).upper()
            # Rotated OCR can read CTS as SLO/SIO and 2010 backwards as 0102.
            # Those signatures are specific to the printed vertical CTS mark.
            if (re.search(r"(?:CTS|SLO|SIO)", cts_text)
                    and re.search(r"(?:20[0-9]{2}|[0-9]{2}02|010[0-9])", cts_text)):
                hints.append("CTS-2010")

            # Full-page OCR can mistake MICR digits for an account number.
            # On this cheque template the printed A/c number is in a compact
            # lower-left box.  Read that box first: the earlier broad crop
            # mixed in surrounding print and shortened leading zeroes.
            account_regions = [
                image.crop((
                    int(width * 0.24), int(height * 0.50),
                    int(width * 0.41), int(height * 0.55),
                )),
                # Fallback for templates whose account box is in a different
                # lower-left position.
                image.crop((
                    int(width * 0.03), int(height * 0.45),
                    int(width * 0.45), int(height * 0.65),
                )),
            ]

            account_texts = []
            for account_region in account_regions:
                for psm in (13, 7, 6, 11):
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
            payee_text = self.pytesseract.image_to_string(
                self._preprocess_pil_image(payee_region), config="--psm 6"
            )
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
