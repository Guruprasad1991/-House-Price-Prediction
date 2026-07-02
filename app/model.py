"""
model.py
────────
Singleton model loader.
Loads the trained model + preprocessor once at startup and exposes
a thread-safe predict() function used by the API endpoints.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from ml.preprocess import ENGINEERED_FEATURES, engineer_features

logger = logging.getLogger(__name__)

MODELS_DIR        = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH        = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH     = MODELS_DIR / "metadata.json"


class ModelNotReadyError(RuntimeError):
    """Raised when the model artifacts have not been trained / are missing."""


class Predictor:
    """
    Wraps the sklearn model + preprocessor.
    Instantiated once and cached via get_predictor().
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise ModelNotReadyError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python -m ml.train` first."
            )
        if not PREPROCESSOR_PATH.exists():
            raise ModelNotReadyError(
                f"Preprocessor not found at {PREPROCESSOR_PATH}. "
                "Run `python -m ml.train` first."
            )

        self._model        = joblib.load(MODEL_PATH)
        self._preprocessor = joblib.load(PREPROCESSOR_PATH)
        self._metadata     = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        logger.info(
            "Loaded model '%s' (R²=%.4f)",
            self._metadata.get("model_name", "unknown"),
            self._metadata.get("metrics", {}).get("r2", float("nan")),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def predict_single(self, raw_features: dict) -> float:
        """
        Predict the median house value for one house.

        Args:
            raw_features: dict with the 8 raw feature keys from HouseFeatures.

        Returns:
            Predicted value in $100k units.
        """
        row = pd.DataFrame([raw_features])
        row = engineer_features(row)
        X   = row[ENGINEERED_FEATURES]
        X_s = self._preprocessor.transform(X)
        return float(self._model.predict(X_s)[0])

    def predict_batch(self, raw_features_list: List[dict]) -> List[float]:
        """Predict for a list of houses in one vectorised call."""
        df  = pd.DataFrame(raw_features_list)
        df  = engineer_features(df)
        X   = df[ENGINEERED_FEATURES]
        X_s = self._preprocessor.transform(X)
        return self._model.predict(X_s).tolist()

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def model_name(self) -> str:
        return self._metadata.get("model_name", "unknown")


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """
    Return the singleton Predictor, loading it on first call.
    lru_cache ensures the heavy joblib.load() happens only once.
    """
    return Predictor()
