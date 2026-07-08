from functools import lru_cache

from fastapi import APIRouter, File, UploadFile

from ..config import get_settings
from ..services.pdf_loader import PDFLoader
from ..services.pipeline import DocumentPipeline
from ..utils.exceptions import PDFValidationError, PipelineError
from ..utils.logger import Logger

router = APIRouter(tags=["Document Extraction"])
logger = Logger.get_logger()


@lru_cache
def get_pipeline() -> DocumentPipeline:
    return DocumentPipeline()


@lru_cache
def get_pdf_loader() -> PDFLoader:
    return PDFLoader(get_settings())


def _error_response(message: str, code: str, details: dict | None = None) -> dict:
    return {
        "success": False,
        "document_type": get_settings().document_type,
        "pages": 0,
        "fields": {},
        "processing_time_ms": 0,
        "errors": [{
            "code": code,
            "message": message,
            "details": details or {}
        }]
    }


@router.post("/extract")
async def extract_document(file: UploadFile = File(...)) -> dict:
    try:
        uploaded_pdf, original_filename = await get_pdf_loader().save_upload(file)
        pipeline = get_pipeline()
        return await pipeline.process(uploaded_pdf, original_filename)
    except PDFValidationError as exc:
        return _error_response(exc.message, exc.code, exc.details)
    except PipelineError as exc:
        return _error_response(exc.message, exc.code, exc.details)
    except Exception as exc:
        logger.exception("Unhandled extraction error")
        return _error_response("Unexpected document extraction failure", "UNEXPECTED_ERROR", {"error": str(exc)})
