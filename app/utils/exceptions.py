class ImageValidationError(Exception):

    def __init__(self, message):

        self.message = message

        super().__init__(self.message)



class DetectionError(Exception):

    def __init__(self, message):

        self.message = message

        super().__init__(self.message)



class OCRError(Exception):

    def __init__(self, message):

        self.message = message

        super().__init__(self.message)


class PipelineError(Exception):
    def __init__(self, message, code="PIPELINE_ERROR", details=None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class PDFValidationError(PipelineError):
    def __init__(self, message, details=None):
        super().__init__(message, "INVALID_PDF", details)
