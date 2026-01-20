from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from mrz_backend.main import app as mrz_app
from face_backend.main import app as face_app

app = FastAPI(title="Unified Access Control API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/mrz", mrz_app)
app.mount("/api/face", face_app)

@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------
# Serve Frontend (Vite dist/)
# -----------------------------
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist"))
ASSETS_DIR = os.path.join(FRONTEND_DIST, "assets")
INDEX_HTML = os.path.join(FRONTEND_DIST, "index.html")

if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

if os.path.isfile(INDEX_HTML):
    @app.get("/")
    def serve_index():
        return FileResponse(INDEX_HTML)

    # SPA fallback: any non-API route should return index.html
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str, request: Request):
        # Do NOT hijack API endpoints
        if full_path.startswith("api/") or full_path in {"docs", "openapi.json", "health"}:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(INDEX_HTML)
