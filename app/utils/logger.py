import logging
from ..config import get_settings


class Logger:

    @staticmethod
    def get_logger():
        settings = get_settings()
        settings.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=settings.log_dir / "document_api.log",
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        return logging.getLogger("DocumentAPI")
