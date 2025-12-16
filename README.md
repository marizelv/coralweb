# Vercel Image Upload Prediction (TensorFlow/Keras)

## What this contains
- `api/predict.py` — FastAPI serverless function that accepts an image upload and returns predictions
- `public/index.html` — Simple frontend to upload images and display prediction results
- `model/` — Put your `model.h5` here (NOT included in repo for size/privacy)
- `requirements.txt` — Python dependencies
- `vercel.json` — Vercel configuration

## How to use
1. Put your Keras model at `model/model.h5`.
2. Edit `api/predict.py` to update `CLASS_NAMES` and `INPUT_SIZE` if needed.
3. Install Vercel CLI:
   ```bash
   npm i -g vercel
   vercel
   ```
4. On deployment, Vercel will serve the frontend at `/` and the prediction endpoint at `/predict`.

## Notes & Tips
- Cold start: the model loads when the serverless function is first invoked. Large TensorFlow models may cause cold-start latency or exceed function memory/time limits. Consider smaller models, TF Lite, or hosting the model on a separate server if needed.
- If your model is PyTorch, ONNX or uses different preprocessing (channels_first), modify `api/predict.py` accordingly.
- If you hit memory or execution time limits on Vercel, consider using a separate inference server (e.g., Google Cloud Run, AWS Lambda with larger resources, or a small VM).
