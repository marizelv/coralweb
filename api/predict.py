import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import traceback
import os

app = FastAPI()

MODEL_PATH = os.path.join("model", "model.h5")

CLASS_NAMES = [
    'ACER','APAL','CNAT','DANT','DSTR','GORG',
    'MALC','MCAV','MEA','MONT','PALY','SPO','SSD','TUN'
]

INPUT_SIZE = (256, 256)

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

MODEL = load_model()

def preprocess_image(image: Image.Image):
    image = image.resize(INPUT_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        x = preprocess_image(img)
        preds = MODEL.predict(x)

        idx = int(np.argmax(preds))
        return {
            "class_index": idx,
            "class_name": CLASS_NAMES[idx],
            "confidence": float(preds[0][idx])
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
