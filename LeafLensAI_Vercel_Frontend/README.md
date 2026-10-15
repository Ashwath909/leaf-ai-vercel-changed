# LeafLens AI — Vercel Frontend

This folder is for the Vercel deployment only. It contains a static web UI and lightweight Python proxy endpoints. It intentionally does not include TensorFlow or the model files.

## Deploy
1. Upload the contents of this folder (index.html, app.js, styles.css, vercel.json, api/) to a GitHub repository.
2. Import that repository in Vercel. Set Root Directory to the repository root (the folder containing `vercel.json` and `api/`).
3. Do not add TensorFlow, tensorflow-cpu, PyTorch, or other model packages to this frontend's dependencies. There should be no root requirements.txt for TensorFlow.
4. Deploy. Open `/api` to check the lightweight API health endpoint.
5. To enable predictions, deploy `LeafLensAI_Python_Backend` separately to a Python hosting service. In Vercel Project Settings → Environment Variables, add `LEAFLENS_API_URL` with the full prediction endpoint, e.g. `https://YOUR-BACKEND-DOMAIN/predict`, then redeploy.

## Important
The frontend can deploy without a trained model. Real predictions require a dataset, a trained model, and the separate Python backend.
