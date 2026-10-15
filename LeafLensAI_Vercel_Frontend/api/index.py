from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            "service": "LeafLens AI Vercel API",
            "status": "ok",
            "prediction_route": "/api/predict",
            "configured": bool(__import__("os").environ.get("LEAFLENS_API_URL")),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
