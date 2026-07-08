from fastapi import FastAPI
from .routes.cheque import router

app = FastAPI(
    title="Cheque Processing API",
    description="Cheque OCR and validation using FastAPI, PyMuPDF, YOLOv8, and PaddleOCR",
    version="2.0"
)

app.include_router(router)

@app.get("/")
def home():

    return {
        "status": True,
        "message": "Cheque Processing API Running",
        "upload_endpoint": "POST /cheque/upload"
    }
