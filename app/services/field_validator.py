from datetime import date
from decimal import Decimal, InvalidOperation

from ..config import Settings


class FieldValidator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self, fields: dict) -> tuple[dict, list[dict]]:
        errors: list[dict] = []

        for required_field in self.settings.required_fields:
            if required_field not in fields or not fields[required_field]["value"]:
                errors.append({
                    "field": required_field,
                    "code": "MISSING_REQUIRED_FIELD",
                    "message": f"Required field '{required_field}' is missing"
                })

        for field_name, field in fields.items():
            field_errors: list[str] = []

            if field["confidence"] < self.settings.ocr_confidence_threshold:
                field_errors.append("LOW_CONFIDENCE")

            if field_name in self.settings.date_fields and not self._is_iso_date(field["value"]):
                field_errors.append("INVALID_DATE")

            if field_name in self.settings.numeric_fields and not self._is_number(field["value"]):
                field_errors.append("INVALID_NUMBER")

            field["errors"] = field_errors
            field["valid"] = not field_errors and bool(field["value"])

            for error_code in field_errors:
                errors.append({
                    "field": field_name,
                    "code": error_code,
                    "message": self._message_for_error(field_name, error_code)
                })

        return fields, errors

    @staticmethod
    def _is_iso_date(value: str) -> bool:
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_number(value: str) -> bool:
        try:
            Decimal(value.replace(",", ""))
            return True
        except (InvalidOperation, AttributeError):
            return False

    @staticmethod
    def _message_for_error(field_name: str, error_code: str) -> str:
        messages = {
            "LOW_CONFIDENCE": f"Field '{field_name}' is below the configured confidence threshold",
            "INVALID_DATE": f"Field '{field_name}' must use ISO date format YYYY-MM-DD",
            "INVALID_NUMBER": f"Field '{field_name}' must contain a valid number"
        }
        return messages.get(error_code, f"Field '{field_name}' is invalid")
