import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import traceback
import os

app = FastAPI()

# ---- USER: place your Keras model file at model/model.h5 in the project root ----
MODEL_PATH = os.path.join("model", "model.h5")

# default class names placeholder (edit to your actual classes)
CLASS_NAMES = ['ACER', 'APAL', 'CNAT', 'DANT', 'DSTR', 'GORG', 'MALC', 'MCAV', 'MEA', 'MONT', 'PALY', 'SPO', 'SSD', 'TUN']

# expected input size (width, height)
INPUT_SIZE = (256, 256)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Upload your model to that path.")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# load model at cold start
try:
    MODEL = load_model()
except Exception as e:
    # keep MODEL as None and return helpful error on requests
    MODEL = None
    LOAD_ERROR = str(e) + "\n" + traceback.format_exc()

def preprocess_image(image: Image.Image):
    # Resize, convert to array, scale to [0,1], expand batch dim
    image = image.resize(INPUT_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0
    # If model expects channels_first, user must modify this function.
    if arr.ndim == 2:
        # grayscale -> RGB
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        # RGBA -> RGB
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/")
async def root():
    return {"status": "ok", "message": "Image prediction API. POST an image file to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model not loaded", "details": LOAD_ERROR}
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        input_arr = preprocess_image(img)
        preds = MODEL.predict(input_arr)
        # handle models that output a single value or vector
        preds_list = preds.tolist()
        if hasattr(preds, 'shape') and len(preds.shape) == 2 and preds.shape[1] > 1:
            class_idx = int(np.argmax(preds, axis=1)[0])
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
        else:
            # binary/single output -> threshold at 0.5
            val = float(preds.flatten()[0])
            class_idx = 1 if val >= 0.5 else 0
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
        return {
            "predicted_class_index": class_idx,
            "predicted_class_name": class_name,
            "raw_output": preds_list
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
