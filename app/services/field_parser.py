import re


class FieldParser:
    def parse(self, region_results: list[dict]) -> dict:
        fields: dict[str, dict] = {}

        for region in region_results:
            field_name = self._normalize_field_name(region["label"])
            ocr = region["ocr"]
            confidence = self._combined_confidence(
                region["detection_confidence"],
                ocr["confidence"]
            )

            fields[field_name] = {
                "value": self._clean_text(ocr["text"]),
                "confidence": confidence,
                "yolo_confidence": region["detection_confidence"],
                "ocr_confidence": ocr["confidence"],
                "valid": False,
                "errors": []
            }

        return fields

    @staticmethod
    def _normalize_field_name(label: str) -> str:
        field = label.strip().lower()
        field = re.sub(r"[^a-z0-9]+", "_", field)
        return field.strip("_") or "unknown"

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _combined_confidence(yolo_confidence: float, ocr_confidence: float) -> float:
        if not yolo_confidence or not ocr_confidence:
            return 0.0
        return round((yolo_confidence + ocr_confidence) / 2, 4)
