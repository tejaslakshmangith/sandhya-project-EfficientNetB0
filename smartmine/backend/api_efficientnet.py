import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-model"))
from inference_efficientnet import predict_image

app = FastAPI(title="SmartMine EfficientNetB0 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "SmartMine EfficientNetB0 API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff"}
    raw_ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "jpg"
    ext = raw_ext if raw_ext in ALLOWED_EXTENSIONS else "jpg"
    temp_filename = f"{UPLOAD_DIR}/{uuid.uuid4()}.{ext}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_image(temp_filename)
        return result

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
