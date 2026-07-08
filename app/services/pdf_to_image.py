from pathlib import Path
import fitz

from ..config import Settings


class PDFToImageConverter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def convert(self, pdf_path: Path, output_dir: Path) -> list[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        zoom = self.settings.pdf_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[dict] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_index}.jpg"
                pixmap.save(image_path)
                pages.append({
                    "page_number": page_index,
                    "image_path": str(image_path),
                    "width": pixmap.width,
                    "height": pixmap.height,
                    "dpi": self.settings.pdf_dpi
                })

        return pages
