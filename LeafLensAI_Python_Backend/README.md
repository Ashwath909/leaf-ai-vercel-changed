# LeafLens AI — Python inference backend

This FastAPI service runs the TensorFlow model. Host it on a Python-capable service (for example, Render or another container/Python host); Vercel hosts the frontend and proxies `/api/predict` requests to this service.

## Train the model

1. Create `dataset/<class_name>/image.jpg` folders under this `backend` directory.
2. Install the packages from `requirements.txt` in a Python 3.11 environment.
3. Run `python train.py` from this directory.
4. The training script writes `models/leaf_lens_model.keras` and `models/class_names.txt`.
5. Ensure both model files are present in the deployed backend. Do not commit a large model to GitHub if your hosting provider has a file-size limit; use the provider's supported persistent file storage instead.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health: `http://localhost:8000/health`
Prediction: `POST http://localhost:8000/predict` as `multipart/form-data`, field name `image`.

The API will return HTTP 503 until the trained model and class names exist.
