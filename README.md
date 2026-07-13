# Cheque Processing API

FastAPI-based cheque OCR service for extracting and validating cheque fields from PDF/image uploads.

## Features

- Upload cheque PDFs or images.
- Render PDF pages using PyMuPDF.
- Extract cheque fields:
  - account number
  - IFSC
  - customer/account-holder name
  - cheque number from MICR line
  - CTS mark/code
  - signature presence
- MICR cheque-number extraction using `ultimateMICR-SDK` when available.
- Tesseract/PaddleOCR fallback for text extraction.
- Multi-page cheque-book sequence correction for OCR outliers.

## Requirements

- Windows
- Python 3.13
- Tesseract OCR installed at one of:
  - `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - path configured through `TESSERACT_CMD`
- Python dependencies from `requirements.txt`
- Optional but recommended: `third_party/ultimateMICR-SDK`

Important: `ultimateMICR-SDK` may show a non-commercial/unlicensed warning. Check its license before using in production/commercial systems.

## Setup

From this folder:

```bash
cd C:\Users\tejeshwarS\Downloads\Cheque_api-office2_updated\Cheque_api-office2\chequeupdated
python -m pip install -r requirements.txt
```

If Tesseract is not in PATH, set:

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

If using a custom ultimateMICR recognizer path:

```bash
set ULTIMATE_MICR_RECOGNIZER=C:\path\to\recognizer.exe
```

## Run API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET http://127.0.0.1:8000/
```

Upload endpoint:

```text
POST http://127.0.0.1:8000/cheque/upload
```

Request body: `multipart/form-data`

Field:

```text
file: PDF/image cheque file
```

## Example Response

```json
{
  "status": true,
  "message": "Cheque processed with validation failures",
  "validation": {
    "total_cheques": 2,
    "valid_cheques": 0,
    "invalid_cheques": 2
  },
  "data": {
    "no_of_cheques": 2,
    "cheques": [
      {
        "account_no": "694010110009816",
        "ifsc": "BKID0006940",
        "customer_name": "SUAENDAA KUMAR SO FAM",
        "cheque_no": "041500",
        "cts": "",
        "signature": true,
        "validations": {
          "cts": false,
          "account_no": true,
          "cheque_no": true,
          "ifsc": true,
          "signature": true,
          "customer_name": true,
          "is_valid": false
        },
        "valid": false
      }
    ]
  }
}
```

## Configuration

Environment variables supported by `app/config.py`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `Document Processing API` | Application name |
| `DOCUMENT_TYPE` | `document` | Document type label |
| `TEMP_DIR` | `app/temp` | Temporary files folder |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | YOLO model path |
| `MAX_PDF_SIZE_MB` | `25` | Max upload size |
| `PDF_DPI` | `300` | PDF render DPI |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.40` | YOLO confidence |
| `USE_CHEQUE_DETECTOR` | `false` | Enable only with a cheque-trained YOLO model |
| `OCR_CONFIDENCE_THRESHOLD` | `0.70` | OCR confidence threshold |
| `TESSERACT_CMD` | unset | Tesseract executable path |
| `ULTIMATE_MICR_RECOGNIZER` | bundled SDK path | Custom ultimateMICR recognizer path |

## Testing

Run unit tests:

```bash
python -m unittest discover -s tests
```

Compile check:

```bash
python -m py_compile app\routes\cheque.py app\services\ocr.py app\services\parser.py
```

## Notes

- Restart the API server after code changes. FastAPI may keep old cached OCR objects while the server is running.
- If port `8000` is already busy, stop the old Python process or run on another port:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

- Cheque number extraction follows the MICR rule: the cheque number is the first six-digit numeric block from the left side of the MICR line.
- Do not treat account number, routing number, transaction code, or last MICR block as the cheque number.
