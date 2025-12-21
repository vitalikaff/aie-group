from __future__ import annotations

import io
import time
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from . import __version__ as eda_version
from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="eda-cli API",
    description="HTTP-сервис качества датасетов поверх eda-cli (FastAPI).",
    version=eda_version,
)


# --------------------------
# Pydantic модели запроса/ответа
# --------------------------
class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None
    latency_ms: float
    flags: Dict[str, object]
    dataset_shape: Optional[Tuple[int, int]] = None
    request_id: Optional[str] = None


# --------------------------
# Утилиты
# --------------------------
def _compute_simple_quality_from_request(req: QualityRequest) -> Dict[str, object]:
    """
    Простая эвристика качества на основе агрегированных признаков.
    Возвращает словарь flags и итоговый quality_score (0..1).
    """
    flags: Dict[str, object] = {}
    score = 1.0 - req.max_missing_share

    # простые эвристики (подглядели в core.compute_quality_flags)
    flags["too_few_rows"] = req.n_rows < 100
    flags["too_many_columns"] = req.n_cols > 100
    flags["max_missing_share"] = req.max_missing_share
    flags["too_many_missing"] = req.max_missing_share > 0.5

    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["too_many_missing"]:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = float(score)
    return flags


# --------------------------
# Эндпоинты
# --------------------------
@app.get("/health", tags=["service"])
def health():
    """
    Простая проверка состояния сервиса.
    """
    return {"status": "ok", "service": "eda-cli-api", "version": eda_version}


@app.post("/quality", response_model=QualityResponse, tags=["model"])
def quality(req: QualityRequest):
    """
    Принимает агрегированные признаки (n_rows, n_cols, max_missing_share, ...) и
    возвращает эвристическую оценку качества и флаги.
    """
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Валидация базовая (Pydantic уже проверил типы)
    if req.n_rows < 0 or req.n_cols < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="n_rows/n_cols must be >= 0")

    flags = _compute_simple_quality_from_request(req)
    quality_score = float(flags.get("quality_score", 0.0))

    latency_ms = (time.perf_counter() - start) * 1000.0
    ok_for_model = quality_score >= 0.5

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        message="Computed from aggregated features",
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape=(req.n_rows, req.n_cols),
        request_id=request_id,
    )


@app.post("/quality-from-csv", response_model=QualityResponse, tags=["model"])
async def quality_from_csv(file: UploadFile = File(...)):
    """
    Загружает CSV (multipart/form-data), читает в pandas, выполняет summarize_dataset,
    missing_table и compute_quality_flags и возвращает результат.
    """
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Проверяем content-type на уровне простого правила (не строго обязательно)
    if file.content_type is not None and "csv" not in file.content_type and "text" not in file.content_type:
        # не строгая проверка — просто предупреждение
        pass

    try:
        body = await file.read()
        if not body:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
        # читаем bytes через BytesIO.
        df = pd.read_csv(io.BytesIO(body))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot read CSV: {exc}")

    # Используем ядро eda-cli
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    quality_score = float(flags.get("quality_score", 0.0))

    latency_ms = (time.perf_counter() - start) * 1000.0
    ok_for_model = quality_score >= 0.5

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        message="Computed from CSV using eda-cli core",
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape=(summary.n_rows, summary.n_cols),
        request_id=request_id,
    )


@app.post("/quality-flags-from-csv", tags=["model"])
async def quality_flags_from_csv(file: UploadFile = File(...)):
    """
    Доп. эндпоинт (HW04): возвращает только словарь флагов качества (включая
    ваши новые эвристики из HW03), прочитав CSV.
    Формат ответа:
    { "flags": { ... } }
    """
    try:
        body = await file.read()
        if not body:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
        df = pd.read_csv(io.BytesIO(body))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot read CSV: {exc}")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Возвратим только flags (JSON-совместимый)
    return {"flags": flags}
