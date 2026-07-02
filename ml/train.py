"""
train.py
────────
Train a house-price prediction model and save artifacts to disk.

Usage:
    python -m ml.train                     # train with defaults
    python -m ml.train --model random_forest --n-estimators 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from ml.evaluate import feature_importance_report, regression_report
from ml.preprocess import ENGINEERED_FEATURES, prepare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH       = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH= MODELS_DIR / "preprocessor.joblib"
METADATA_PATH    = MODELS_DIR / "metadata.json"


def get_model(name: str, **kwargs):
    """Return an unfitted sklearn estimator by name."""
    catalogue = {
        "linear":       LinearRegression(),
        "ridge":        Ridge(alpha=kwargs.get("alpha", 1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 4),
            n_jobs=-1,
            random_state=42,
        ),
    }
    if name not in catalogue:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(catalogue)}"
        )
    return catalogue[name]


def train(
    model_name: str = "random_forest",
    test_size: float = 0.2,
    **model_kwargs,
) -> dict:
    """
    Full training run: preprocess → fit → evaluate → save artifacts.

    Returns:
        metrics dict
    """
    # 1. Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare(
        test_size=test_size
    )

    # 2. Select and train the model
    logger.info("Training model: %s", model_name)
    model = get_model(model_name, **model_kwargs)
    model.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = model.predict(X_test)
    metrics = regression_report(y_test, y_pred, label=model_name.replace("_", " ").title())
    feature_importance_report(model, ENGINEERED_FEATURES)

    # 4. Save artifacts
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Model saved    → %s", MODEL_PATH)
    logger.info("Preprocessor  → %s", PREPROCESSOR_PATH)

    # 5. Write metadata (used by the API /health endpoint)
    metadata = {
        "model_name":    model_name,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "feature_names": ENGINEERED_FEATURES,
        "metrics": {
            "mae":  round(metrics["mae"],  6),
            "rmse": round(metrics["rmse"], 6),
            "r2":   round(metrics["r2"],   6),
            "mape": round(metrics["mape"], 4),
        },
        "test_size":     test_size,
        "sklearn_params": model.get_params(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata       → %s", METADATA_PATH)

    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the house-price model.")
    parser.add_argument(
        "--model", default="random_forest",
        choices=["linear", "ridge", "random_forest"],
        help="Which model to train (default: random_forest)"
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth",    type=int, default=None)
    parser.add_argument("--alpha",        type=float, default=1.0,
                        help="Regularisation strength for Ridge")
    parser.add_argument("--test-size",    type=float, default=0.2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    metrics = train(
        model_name=args.model,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        alpha=args.alpha,
    )
    r2 = metrics["r2"]
    if r2 < 0.5:
        logger.error("R² below 0.5 — model quality is poor. Aborting.")
        sys.exit(1)
    logger.info("Training complete. R² = %.4f", r2)
