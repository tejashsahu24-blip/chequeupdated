from .image_quality import ImageQuality
from .detector import ChequeDetector
from .cropper import Cropper
from .ocr import OCRService
from .parser import Parser
from .validator import Validator
import cv2


class ChequeService:

    def __init__(self):

        self.detector = ChequeDetector()

        self.ocr = OCRService()
