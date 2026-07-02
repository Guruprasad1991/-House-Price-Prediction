"""
main.py
───────
FastAPI application — House Price Prediction API.

Endpoints:
    GET  /           → redirect to docs
    GET  /health     → model status + metrics
    POST /predict    → single house prediction
    POST /predict/batch → batch predictions (up to 1000)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse

from app.model import ModelNotReadyError, get_predictor
from app.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request is fast."""
    logger.info("Loading model …")
    try:
        get_predictor()
        logger.info("Model ready.")
    except ModelNotReadyError as exc:
        logger.warning("Model not ready: %s", exc)
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predicts California median house values using a scikit-learn model "
        "trained on the California Housing dataset.\n\n"
        "**Quick start**\n"
        "```bash\n"
        "curl -X POST http://localhost:8000/predict \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"MedInc\":8.3,\"HouseAge\":41,\"AveRooms\":6.98,"
        "\"AveBedrms\":1.02,\"Population\":322,\"AveOccup\":2.56,"
        "\"Latitude\":37.88,\"Longitude\":-122.23}'\n"
        "```"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── OpenAPI schema: force 3.0.3 for Swagger UI compatibility ─────────────────
# FastAPI 0.99+ emits OpenAPI 3.1.0 with `exclusiveMinimum: <number>`.
# Swagger UI < 5.x and some 5.x builds misparse that format.
# This post-processes the schema to OpenAPI 3.0.3 (boolean exclusiveMinimum)
# which every Swagger UI version handles correctly.

def _fix_exclusive_bounds(obj: Any) -> None:
    """Recursively rewrite 3.1 numeric exclusiveMinimum/Maximum to 3.0 style."""
    if isinstance(obj, dict):
        if "exclusiveMinimum" in obj and isinstance(obj["exclusiveMinimum"], (int, float)):
            obj["minimum"] = obj.pop("exclusiveMinimum")
            obj["exclusiveMinimum"] = True
        if "exclusiveMaximum" in obj and isinstance(obj["exclusiveMaximum"], (int, float)):
            obj["maximum"] = obj.pop("exclusiveMaximum")
            obj["exclusiveMaximum"] = True
        for v in obj.values():
            _fix_exclusive_bounds(v)
    elif isinstance(obj, list):
        for item in obj:
            _fix_exclusive_bounds(item)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"   # downgrade version string
    _fix_exclusive_bounds(schema)
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi  # type: ignore[method-assign]


# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser to interactive docs."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
    tags=["Meta"],
)
async def health():
    """
    Returns the model's training metrics and readiness status.
    Use this to verify the service is up and the model is loaded.
    """
    try:
        predictor = get_predictor()
        meta = predictor.metadata
        return HealthResponse(
            status="ok",
            model_name=meta.get("model_name", "unknown"),
            trained_at=meta.get("trained_at", "unknown"),
            metrics=meta.get("metrics", {}),
            feature_names=meta.get("feature_names", []),
        )
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict price for one house",
    tags=["Prediction"],
)
async def predict(features: HouseFeatures):
    """
    Predict the median house value for a single house.

    All eight raw features are required. The API internally engineers
    three additional features before running inference.
    """
    try:
        predictor = get_predictor()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    raw = features.model_dump()
    try:
        price_100k = predictor.predict_single(raw)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    price_usd = price_100k * 100_000
    logger.info(
        "Prediction: $%,.0f  (lat=%.2f, lon=%.2f, MedInc=%.2f)",
        price_usd, raw["Latitude"], raw["Longitude"], raw["MedInc"],
    )

    return PredictionResponse(
        predicted_price_usd=round(price_usd, 2),
        predicted_price_100k=round(price_100k, 4),
        model_name=predictor.model_name,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict prices for multiple houses",
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictRequest):
    """
    Predict median house values for a list of up to 1,000 houses
    in a single vectorised call — much faster than looping /predict.
    """
    try:
        predictor = get_predictor()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    raw_list = [h.model_dump() for h in request.houses]
    try:
        prices_100k = predictor.predict_batch(raw_list)
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

    predictions = [
        PredictionResponse(
            predicted_price_usd=round(p * 100_000, 2),
            predicted_price_100k=round(p, 4),
            model_name=predictor.model_name,
        )
        for p in prices_100k
    ]
    logger.info("Batch prediction: %d houses", len(predictions))
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
