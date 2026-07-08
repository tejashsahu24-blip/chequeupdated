from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from ..config import Settings
from ..utils.exceptions import PipelineError


class PDFLoader:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def save_upload(self, file: UploadFile) -> tuple[Path, str]:
        original_filename = Path(file.filename or "").name
        if not original_filename:
            raise PipelineError("File name is required", "FILE_NAME_REQUIRED")

        if Path(original_filename).suffix.lower() != ".pdf":
            raise PipelineError("Only PDF files are allowed", "UNSUPPORTED_FILE_TYPE")

        request_dir = self.settings.temp_dir / "incoming" / uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=True)
        destination = request_dir / original_filename

        with destination.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        return destination, original_filename
