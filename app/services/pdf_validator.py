from pathlib import Path
import fitz

from ..config import Settings
from ..utils.exceptions import PDFValidationError


class PDFValidator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self, pdf_path: Path) -> dict:
        if not pdf_path.exists():
            raise PDFValidationError("Uploaded file was not saved")

        if pdf_path.suffix.lower() != ".pdf":
            raise PDFValidationError("Only PDF files are allowed")

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb <= 0:
            raise PDFValidationError("Uploaded PDF is empty")

        if size_mb > self.settings.max_pdf_size_mb:
            raise PDFValidationError(
                f"PDF size exceeds {self.settings.max_pdf_size_mb} MB",
                {"size_mb": round(size_mb, 2)}
            )

        with pdf_path.open("rb") as file:
            header = file.read(5)
            if header != b"%PDF-":
                raise PDFValidationError("Uploaded file is not a valid PDF")

        try:
            with fitz.open(pdf_path) as document:
                page_count = document.page_count
                if page_count == 0:
                    raise PDFValidationError("PDF does not contain any pages")
        except PDFValidationError:
            raise
        except Exception as exc:
            raise PDFValidationError("PDF could not be opened", {"error": str(exc)}) from exc

        return {
            "filename": pdf_path.name,
            "size_mb": round(size_mb, 2),
            "pages": page_count
        }
