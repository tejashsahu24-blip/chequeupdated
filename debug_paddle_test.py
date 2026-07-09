from paddleocr import PaddleOCR
from pathlib import Path

sample = Path('app/uploads/page_0.jpg')
print('sample exists', sample.exists())
try:
    ocr = PaddleOCR(
        lang='en',
        device='cpu',
        enable_mkldnn=False,
        enable_cinn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print('created')
    res = ocr.predict(str(sample))
    print('result type', type(res))
    print('result len', len(res) if res else 0)
    print(res[:3])
except Exception as exc:
    print('error', type(exc).__name__, exc)
    import traceback
    traceback.print_exc()