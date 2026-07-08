from pathlib import Path
from time import perf_counter
from uuid import uuid4
import asyncio
import re
import shutil

from ..config import Settings, get_settings
from .document_cropper import DocumentCropper
from .field_parser import FieldParser
from .field_validator import FieldValidator
from .ocr_engine import OCREngine
from .pdf_to_image import PDFToImageConverter
from .pdf_validator import PDFValidator
from .yolo_detector import YOLODetector
from ..utils.exceptions import PipelineError
from ..utils.logger import Logger


class DocumentPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.logger = Logger.get_logger()
        self.pdf_validator = PDFValidator(self.settings)
        self.pdf_converter = PDFToImageConverter(self.settings)
        self.detector = YOLODetector(self.settings)
        self.cropper = DocumentCropper()
        self.ocr = OCREngine()
        self.parser = FieldParser()
        self.validator = FieldValidator(self.settings)

    async def process(self, source_pdf: Path, original_filename: str) -> dict:
        started_at = perf_counter()
        request_id = uuid4().hex
        work_dir = self.settings.temp_dir / request_id
        upload_dir = work_dir / "uploads"
        pages_dir = work_dir / "pages"
        crops_dir = work_dir / "crops"
        ocr_dir = work_dir / "ocr"

        for directory in (upload_dir, pages_dir, crops_dir, ocr_dir):
            directory.mkdir(parents=True, exist_ok=True)

        pdf_path = upload_dir / original_filename
        shutil.copy2(source_pdf, pdf_path)
        errors: list[dict] = []
        page_results: list[dict] = []
        region_results: list[dict] = []

        try:
            pdf_info = await asyncio.to_thread(self.pdf_validator.validate, pdf_path)
            page_images = await asyncio.to_thread(self.pdf_converter.convert, pdf_path, pages_dir)

            for page in page_images:
                page_number = page["page_number"]
                image_path = Path(page["image_path"])
                page_error_count = len(errors)

                try:
                    detections = await asyncio.to_thread(self.detector.detect, image_path)
                    page_crop_dir = crops_dir / f"page_{page_number}"
                    crops = await asyncio.to_thread(self.cropper.crop, image_path, detections, page_crop_dir)

                    page_regions = []
                    for crop_index, crop in enumerate(crops, start=1):
                        if crop["error"]:
                            errors.append({
                                "page": page_number,
                                "field": crop["label"],
                                "code": "CROP_FAILED",
                                "message": crop["error"]
                            })
                            continue

                        crop_path = Path(crop["path"])
                        ocr_result = await asyncio.to_thread(self.ocr.read, crop_path)
                        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", crop["label"]).strip("_") or "field"
                        ocr_path = ocr_dir / f"page_{page_number}_{crop_index}_{safe_label}.txt"
                        ocr_path.write_text(ocr_result["text"], encoding="utf-8")

                        region = {
                            "page": page_number,
                            "label": crop["label"],
                            "box": crop["box"],
                            "detection_confidence": crop["confidence"],
                            "crop_path": crop["path"],
                            "ocr_path": str(ocr_path),
                            "ocr": ocr_result
                        }
                        page_regions.append(region)
                        region_results.append(region)

                    page_results.append({
                        **page,
                        "detections": detections,
                        "regions": page_regions,
                        "errors": errors[page_error_count:]
                    })
                except Exception as exc:
                    self.logger.exception("Page processing failed")
                    errors.append({
                        "page": page_number,
                        "code": "PAGE_PROCESSING_FAILED",
                        "message": str(exc)
                    })
                    page_results.append({
                        **page,
                        "detections": [],
                        "regions": [],
                        "errors": errors[page_error_count:]
                    })

            fields = self.parser.parse(region_results)
            fields, validation_errors = self.validator.validate(fields)
            errors.extend(validation_errors)

            processing_time_ms = int((perf_counter() - started_at) * 1000)
            return {
                "success": not errors,
                "document_type": self.settings.document_type,
                "pages": pdf_info["pages"],
                "fields": fields,
                "processing_time_ms": processing_time_ms,
                "errors": errors,
                "meta": {
                    "request_id": request_id,
                    "filename": original_filename,
                    "pdf": pdf_info,
                    "temp_dir": str(work_dir),
                    "total_regions": len(region_results)
                },
                "page_results": page_results
            }
        except PipelineError:
            raise
        except Exception as exc:
            self.logger.exception("Document pipeline failed")
            raise PipelineError("Document processing failed", "PIPELINE_FAILED", {"error": str(exc)}) from exc
