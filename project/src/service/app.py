"""FastAPI-приложение: эндпоинты /health и /predict."""
from __future__ import annotations

import base64
import io
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from ..config import Config, load_config
from ..inference import PredictionResult, load_model, predict
from ..logging_config import get_logger

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    model_active: str
    architecture: str
    encoder: str | None
    description: str
    device: str
    image_size: list[int]
    threshold: float


class PredictResponse(BaseModel):
    polyp_area_ratio: float
    threshold: float
    image_size: list[int]
    mask_png_base64: str
    elapsed_ms: float


def _encode_mask_png(mask: np.ndarray) -> str:
    """Кодирует бинарную маску (0/1) в PNG base64."""
    vis = (mask.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", vis)
    if not ok:
        raise RuntimeError("Не удалось закодировать маску в PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Bytes -> RGB numpy."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Невалидное изображение: {exc}") from exc


def create_app(config: Optional[Config] = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(
        title="Polyp Segmentation Service",
        description="Сервис сегментации полипов на эндоскопических изображениях (Kvasir-SEG, SwinUNet).",
        version="0.1.0",
    )

    state = {"model": None, "device": None, "cfg": cfg}

    @app.on_event("startup")
    def _startup() -> None:
        logger.info("Starting service, loading model...")
        model, device, cfg_loaded = load_model(cfg)
        state["model"] = model
        state["device"] = device
        state["cfg"] = cfg_loaded
        logger.info("Model ready on %s", device)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        c: Config = state["cfg"]
        spec = c.get_model_spec()
        return HealthResponse(
            status="ok" if state["model"] is not None else "loading",
            model_active=spec.name,
            architecture=spec.architecture,
            encoder=spec.encoder_name,
            description=spec.description,
            device=str(state["device"]) if state["device"] else "unknown",
            image_size=list(c.inference.image_size),
            threshold=c.inference.threshold,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict_endpoint(file: UploadFile = File(...)) -> PredictResponse:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Модель ещё не загружена")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Пустой файл")

        image_rgb = _read_image(content)
        logger.info("Received image '%s' shape=%s", file.filename, image_rgb.shape)

        start = time.perf_counter()
        result: PredictionResult = predict(state["model"], image_rgb, state["cfg"], state["device"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "Prediction done: area=%.4f, threshold=%.2f, %.1f ms",
            result.polyp_area_ratio, result.threshold, elapsed_ms,
        )

        return PredictResponse(
            polyp_area_ratio=result.polyp_area_ratio,
            threshold=result.threshold,
            image_size=list(result.image_size),
            mask_png_base64=_encode_mask_png(result.mask),
            elapsed_ms=round(elapsed_ms, 2),
        )

    return app


app = create_app()


def main() -> None:
    """Точка входа `python -m src.service`."""
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "src.service.app:app",
        host=cfg.service.host,
        port=cfg.service.port,
        log_level=cfg.service.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
