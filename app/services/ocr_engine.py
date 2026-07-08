from pathlib import Path
from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self):
        self.engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    def read(self, image_path: Path) -> dict:
        result = self.engine.ocr(str(image_path))
        lines: list[dict] = []

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = float(line[1][1])
                lines.append({
                    "text": text,
                    "confidence": round(confidence, 4)
                })

        average_confidence = (
            sum(line["confidence"] for line in lines) / len(lines)
            if lines else 0.0
        )

        return {
            "text": " ".join(line["text"] for line in lines).strip(),
            "confidence": round(average_confidence, 4),
            "lines": lines
        }
