from functools import lru_cache

from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from uuid import uuid4
import shutil
import cv2
import fitz
import numpy as np
from PIL import Image as PILImage

from ..config import get_settings
from ..services.image_quality import ImageQuality
from ..services.detector import ChequeDetector
from ..services.cropper import Cropper
from ..services.ocr import OCRService
from ..services.parser import Parser
from ..services.validator import Validator
from ..services.signature_checker import SignatureChecker

router = APIRouter(prefix="/cheque", tags=["Cheque OCR"])

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}


def _response(status, message, cheques, validations=None):
    return {
        "status": status,
        "message": message,
        "validation": validations if validations else {},
        "data": {
            "no_of_cheques": len(cheques),
            "cheques": cheques
        }
    }


@lru_cache(maxsize=1)
def _get_detector():
    """Load the model once per API worker, not once per uploaded file."""
    return ChequeDetector()


@lru_cache(maxsize=1)
def _get_ocr():
    """OCR startup is expensive, so retain the initialized backend."""
    return OCRService()


def _render_pdf_pages(pdf_path):

    print("Opening:", pdf_path)

    settings = get_settings()
    pdf = fitz.open(str(pdf_path))

    print("Pages:", len(pdf))

    rendered = []
    texts = []

    for i, page in enumerate(pdf):

        pix = page.get_pixmap(dpi=settings.pdf_dpi, alpha=False)
        img = pdf_path.with_name(f"{pdf_path.stem}_page_{i}.png")
        pix.save(str(img))
        rendered.append(img)

        texts.append(Parser.clean_text(page.get_text()))

    pdf.close()
    print("Rendered pages:", len(rendered))
    print("Extracted texts:", len(texts))

    return rendered, texts


def _extract_fields(text):
    return Parser.extract_fields(Parser.clean_text(text))


def _validate_fields(fields, signature_status):
    validations = {
        "cts": Validator.validate_cts(fields.get("cts")),
        "account_no": Validator.validate_account(fields.get("account_number")),
        "cheque_no": Validator.validate_cheque_number(fields.get("cheque_number")),
        "ifsc": Validator.validate_ifsc(fields.get("ifsc")),
        "signature": Validator.validate_signature(signature_status),
        "customer_name": Validator.validate_customer_name(fields.get("customer_name"))
    }
    validations["is_valid"] = all(validations.values())

    return validations


def _build_cheque_result(fields, validations, signature_status):
    # Keep this shape stable even when a field could not be read.  API
    # consumers can therefore always rely on the same JSON keys.
    return {
        "account_no": fields.get("account_number"),
        "ifsc": fields.get("ifsc"),
        "customer_name": fields.get("customer_name"),
        "cheque_no": fields.get("cheque_number"),
        "cts": fields.get("cts"),
        "signature": signature_status,
        "validations": validations,
        "valid": validations.get("is_valid", False)
    }


def _get_page_text(image_path, pdf_texts, page_index, ocr):
    # print(f"Page {page_index}: Extracting text from image: {image_path} and PDF texts: {pdf_texts} and page index: {page_index} and ocr: {ocr}")
    if pdf_texts and page_index - 1 < len(pdf_texts) and pdf_texts[page_index - 1]:
        return pdf_texts[page_index - 1]

    ocr_raw_result = ocr.read_text(str(image_path))
    # print(f"Page {page_index}: OCR Raw Result: {ocr_raw_result}")
    ocr_text, _ = ocr.extract_text(ocr_raw_result)

    if not ocr_text and pdf_texts and page_index - 1 < len(pdf_texts):
        return pdf_texts[page_index - 1]

    return ocr_text


def _get_cheque_text(image_path, pdf_texts, page_index, ocr):
    """Combine page OCR/PDF text with targeted CTS and serial-number reads."""
    page_text = _get_page_text(image_path, pdf_texts, page_index, ocr)
    field_hints = ocr.read_cheque_field_hints(str(image_path))
    print(f"Page Text: {page_text}")
    print(f"Field Hints: {field_hints}")
    # Targeted reads must precede full-page text: otherwise a digit sequence
    # from the MICR line can be selected before the labelled account-box read.
    return _merge_text(field_hints, page_text)


def _merge_text(*values):
    """Retain the strongest available text source for a cheque page."""
    return " ".join(value.strip() for value in values if value and value.strip())


@router.post("/upload")
async def upload_cheque(file: UploadFile = File(...)):

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name:
        return _response(False, "File name is required", [])

    if extension not in SUPPORTED_EXTENSIONS:
        return _response(False, "Only PDF and image files are allowed", [])

    file_path = UPLOAD_DIR / f"{Path(original_name).stem}_{uuid4().hex}{extension}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    size_status, size_message = ImageQuality.check_file_size(file_path)
    if not size_status:
        return _response(False, size_message, [])

    is_pdf = extension == ".pdf"
    image_paths = [file_path]
    pdf_texts = []

    if is_pdf:
        try:
            image_paths, pdf_texts = _render_pdf_pages(file_path)
            print(f"Rendered {len(image_paths)} pages from PDF: {file_path}")
            print(f"Extracted texts: {len(pdf_texts)}")
        except Exception as e:
            return _response(False, f"PDF Error : {str(e)}", [])

        if not image_paths:
            return _response(False, "PDF does not contain any pages", [])

    try:
        ocr = _get_ocr()
    except Exception as exc:
        return _response(False, f"Processing service unavailable: {exc}", [])

    detector = None
    if get_settings().use_cheque_detector:
        try:
            candidate = _get_detector()
            if candidate.supports_cheque_detection:
                detector = candidate
            else:
                print("Cheque detector is disabled because the model has no cheque class.")
        except Exception as exc:
            # OCR can still return all fields, so a detector issue must not
            # turn an otherwise processable cheque into a failed request.
            print(f"Cheque detector unavailable; continuing with OCR: {exc}")

    cheque_results = []

    for page_index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            try:
                pil_image = PILImage.open(str(image_path)).convert("RGB")
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception:
                continue

        resolution_status, _ = ImageQuality.check_resolution(image)
        blur_status, _ = ImageQuality.check_blur(image)
        brightness_status, _ = ImageQuality.check_brightness(image)

        if not (resolution_status and blur_status and brightness_status):
            ocr_text = _get_cheque_text(image_path, pdf_texts, page_index, ocr)
            # print(f"Page {page_index}: OCR Text: {ocr_text}")
            fields = _extract_fields(ocr_text)
            signature_status, _ = SignatureChecker.check_signature(image)
            validations = _validate_fields(fields, signature_status)

            cheque_results.append(
                _build_cheque_result(fields, validations, signature_status)
            )
            continue

        detections = detector.detect(str(image_path)) if detector else []
        cheques = [
            d for d in detections
            if d["class"].lower() == "cheque"
        ]

        if cheques:
            cropped_images = Cropper.crop_fields(
                str(image_path),
                cheques,
                output_prefix=f"{file_path.stem}_page_{page_index}"
            )

            for cheque_index, cheque in enumerate(cheques, start=1):
                crop_key = f"{cheque['class']}_{cheque_index}"
                crop_path = cropped_images.get(crop_key)

                if not crop_path:
                    continue

                ocr_raw_result = ocr.read_text(crop_path)
                ocr_text, _ = ocr.extract_text(ocr_raw_result)

                # A detector crop is useful for image-only PDFs, but OCR can
                # lose edge fields such as the MICR line.  Include the page's
                # embedded text (when present) before parsing the cheque.
                page_text = _get_cheque_text(image_path, pdf_texts, page_index, ocr)
                fields = _extract_fields(_merge_text(ocr_text, page_text))
                crop_image = cv2.imread(crop_path)
                signature_status, _ = SignatureChecker.check_signature(crop_image)
                validations = _validate_fields(fields, signature_status)

                cheque_results.append(
                    _build_cheque_result(fields, validations, signature_status)
                )
        else:
            # Fallback: when no cheque box is detected, OCR the full page and attempt to extract fields.
            ocr_text = _get_cheque_text(image_path, pdf_texts, page_index, ocr)
            fields = _extract_fields(ocr_text)
            signature_status, _ = SignatureChecker.check_signature(image)
            validations = _validate_fields(fields, signature_status)

            cheque_results.append(
                _build_cheque_result(fields, validations, signature_status)
            )

    if not cheque_results:
        return _response(False, "No cheque detected", [])

    all_valid = all(cheque["valid"] for cheque in cheque_results)
    message = "Cheque processed successfully" if all_valid else "Cheque processed with validation failures"

    overall_validation = {
        "total_cheques": len(cheque_results),
        "valid_cheques": sum(1 for c in cheque_results if c["valid"]),
        "invalid_cheques": sum(1 for c in cheque_results if not c["valid"])
    }

    return _response(True, message, cheque_results, overall_validation)
