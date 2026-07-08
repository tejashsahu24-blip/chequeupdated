from paddleocr import PaddleOCR


class OCRService:

    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    def read_text(self, image_path):
        return self.ocr.ocr(image_path)

    def extract_text(self, result):
        text = []
        confidence = []

        try:
            if result and result[0]:
                for line in result[0]:
                    text.append(line[1][0])
                    confidence.append(line[1][1])

        except Exception as e:
            print("OCR Error:", e)

        avg_conf = sum(confidence) / len(confidence) if confidence else 0

        return " ".join(text), round(avg_conf, 2)
