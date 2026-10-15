"""Small Vercel proxy to a separately hosted LeafLens TensorFlow API.

The TensorFlow model is intentionally not bundled into this Vercel function:
TensorFlow's dependencies and model can exceed practical serverless bundle limits.
Set LEAFLENS_API_URL to the backend prediction endpoint, e.g. https://your-api.example.com/predict.
"""
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import os

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        backend_url = os.environ.get("LEAFLENS_API_URL", "").strip()
        if not backend_url:
            self._json(503, {
                "error": "AI backend is not configured yet. Deploy the Python backend and set LEAFLENS_API_URL in Vercel project environment variables."
            })
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self._json(400, {"error": "Send the image as multipart/form-data using the field name 'image'."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"error": "No image was received."})
            return
        if length > 4 * 1024 * 1024:
            self._json(413, {"error": "Image request exceeds the 4 MB app limit."})
            return
        body = self.rfile.read(length)
        request = Request(backend_url, data=body, headers={"Content-Type": content_type, "Accept": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
                status = response.status
                response_type = response.headers.get("Content-Type", "application/json")
            self.send_response(status)
            self.send_header("Content-Type", response_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except HTTPError as exc:
            try:
                payload = exc.read()
            except Exception:
                payload = b'{"error":"AI backend returned an error."}'
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
        except (URLError, TimeoutError, OSError) as exc:
            self._json(502, {"error": "Could not reach the AI backend. Check the backend URL and whether the service is awake."})

    def _json(self, status, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)
