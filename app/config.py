from functools import lru_cache
from pathlib import Path
import os


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    """Application settings loaded from environment variables."""

    base_dir: Path = Path(__file__).resolve().parent
    project_dir: Path = base_dir.parent

    app_name: str = os.getenv("APP_NAME", "Document Processing API")
    document_type: str = os.getenv("DOCUMENT_TYPE", "document")
    temp_dir: Path = Path(os.getenv("TEMP_DIR", str(base_dir / "temp")))
    model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", str(project_dir / "yolov8n.pt")))

    max_pdf_size_mb: float = float(os.getenv("MAX_PDF_SIZE_MB", "25"))
    pdf_dpi: int = int(os.getenv("PDF_DPI", "300"))
    yolo_confidence_threshold: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.40"))
    ocr_confidence_threshold: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    required_fields: list[str] = _csv(os.getenv("REQUIRED_FIELDS", ""))
    date_fields: list[str] = _csv(os.getenv("DATE_FIELDS", "date,invoice_date"))
    numeric_fields: list[str] = _csv(os.getenv("NUMERIC_FIELDS", "amount,total,total_amount"))

    log_dir: Path = Path(os.getenv("LOG_DIR", str(base_dir / "logs")))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings
