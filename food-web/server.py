"""
FastAPI server for ChineseFoodNet.

Endpoints
---------
GET  /                     -> serves static/index.html
GET  /api/health           -> {"status": "ok", ...}
GET  /api/classes          -> all 208 class labels
POST /api/predict          -> multipart upload field 'file' -> top-K predictions

Run (development):
    uvicorn server:app --reload --host 127.0.0.1 --port 8000

Run (production):
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from inference import (
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TOP_K,
    build_preprocess,
    load_model,
    predict_topk,
    read_class_names,
)
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration (env overrides)
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_DIR / "static"

CKPT_PATH = Path(os.environ.get("CKPT_PATH",
                                str(PROJECT_DIR.parent / "outputs" / "convnextv2_base" / "best.pth")))
CLASS_CSV = Path(os.environ.get("CLASS_CSV",
                                str(PROJECT_DIR.parent / "data" / "metadata" / "class_names.csv")))
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
IMG_SIZE = int(os.environ.get("IMG_SIZE", DEFAULT_IMG_SIZE))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", DEFAULT_NUM_CLASSES))
TOP_K = int(os.environ.get("TOP_K", DEFAULT_TOP_K))
DEVICE = os.environ.get("DEVICE")  # None -> auto detect
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 10))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("foodweb")


# --------------------------------------------------------------------------- #
# App + lifespan: load model once at startup
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading class names from %s", CLASS_CSV)
    class_names = read_class_names(CLASS_CSV, num_classes=NUM_CLASSES)

    logger.info("Loading model %s from %s", MODEL_NAME, CKPT_PATH)
    t0 = time.time()
    model, info = load_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        ckpt_path=CKPT_PATH,
        device=DEVICE,
    )
    preprocess = build_preprocess(IMG_SIZE)
    elapsed = time.time() - t0
    logger.info("Model ready in %.2fs: loaded=%d/%d, device=%s",
                elapsed, info["loaded"], info["total"], info["device"])
    if info.get("head_warning"):
        logger.warning("Head warning: %s", info["head_warning"])

    app.state.model = model
    app.state.preprocess = preprocess
    app.state.class_names = class_names
    app.state.model_info = info
    app.state.startup_time = elapsed

    yield

    # Shutdown (nothing to clean up)
    logger.info("Server shutting down")


app = FastAPI(
    title="ChineseFoodNet API",
    description="Image classification for 208 classes of Chinese food.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - permissive by default, narrow this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Static files (frontend)
# --------------------------------------------------------------------------- #
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "ChineseFoodNet API", "see": "/docs"})


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    info = app.state.model_info
    return {
        "status": "ok",
        "model_name": info["model_name"],
        "num_classes": info["num_classes"],
        "device": info["device"],
        "loaded_tensors": f"{info['loaded']}/{info['total']}",
        "startup_seconds": round(app.state.startup_time, 3),
        "head_warning": info.get("head_warning") or None,
    }


@app.get("/api/classes")
async def get_classes():
    """Return the full list of 208 class labels."""
    return {
        "num_classes": len(app.state.class_names),
        "classes": [
            {"class_id": i, "label_zh": zh, "label_en": en}
            for i, (zh, en) in enumerate(app.state.class_names)
        ],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), top_k: int = TOP_K):
    """Accept an image upload and return top-K predictions."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported content type: {file.content_type}. "
                                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}")

    body = await file.read()
    if len(body) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"Upload too large (>{MAX_UPLOAD_MB}MB)")

    try:
        image = Image.open(io.BytesIO(body))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    top_k = max(1, min(int(top_k), len(app.state.class_names)))

    t0 = time.time()
    preds = predict_topk(
        model=app.state.model,
        image=image,
        class_names=app.state.class_names,
        top_k=top_k,
        preprocess=app.state.preprocess,
    )
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "filename": file.filename,
        "size_bytes": len(body),
        "top_k": top_k,
        "elapsed_ms": round(elapsed_ms, 2),
        "predictions": [p.to_dict() for p in preds],
    }
