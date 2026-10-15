from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "leaf_lens_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.txt"
IMG_SIZE = (224, 224)
app = FastAPI(title="LeafLens AI Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
_model = None
_classes = None

@app.get("/")
def health():
    return {"service": "LeafLens AI API", "status": "ok", "model_ready": MODEL_PATH.exists() and CLASS_PATH.exists()}

@app.get("/health")
def health_check():
    return health()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    global _model, _classes
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        raise HTTPException(status_code=503, detail="Trained model is missing. Train the model and add models/leaf_lens_model.keras and models/class_names.txt to the backend.")
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")
    raw = await image.read()
    if len(raw) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 4 MB limit.")
    try:
        pil_image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB").resize(IMG_SIZE)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    if _model is None:
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            _classes = [x.strip() for x in CLASS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load the trained model: {exc}")
    # The training model includes its own Rescaling layer, so feed pixel values in [0, 255].
    arr = np.asarray(pil_image, dtype=np.float32)[None, ...]
    scores = _model.predict(arr, verbose=0)[0]
    index = int(np.argmax(scores))
    label = _classes[index] if index < len(_classes) else f"Class {index}"
    return {"label": label, "confidence": round(float(scores[index]) * 100, 2), "classes": _classes}
