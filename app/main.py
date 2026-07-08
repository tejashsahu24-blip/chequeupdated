from fastapi import FastAPI
from .routes.cheque import router
from .routes.upload import router as upload_router

app = FastAPI(
    title="Document Processing API",
    description="PDF extraction using FastAPI, PyMuPDF, YOLOv8, and PaddleOCR",
    version="2.0"
)

app.include_router(router)


@app.get("/")
def home():

    return {
        "status": True,
        "message": "Document Processing API Running",
        "extract_endpoint": "POST /extract"
    }
