"""
preprocess.py
─────────────
Data loading, feature engineering, train/test split, and scaling.
All steps are encapsulated in a single Pipeline object so the same
transformations apply to training data AND to live inference payloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Feature names as loaded from the dataset ──────────────────────────────────
RAW_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

# Features added by engineer_features()
ENGINEERED_FEATURES = RAW_FEATURES + [
    "RoomsPerHousehold",
    "BedroomRatio",
    "PopulationPerHousehold",
]

TARGET = "MedHouseVal"


def load_data() -> pd.DataFrame:
    """
    Load the California Housing dataset and return it as a DataFrame.
    Prices are in units of $100,000 (as shipped by sklearn).
    """
    logger.info("Loading California Housing dataset …")
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.rename(columns={"MedHouseVal": TARGET})
    logger.info("Dataset loaded: %d rows, %d columns", *df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three derived features that carry useful signal.

    RoomsPerHousehold     – average rooms per occupied household
    BedroomRatio          – fraction of rooms that are bedrooms
    PopulationPerHousehold– average people per household
    """
    df = df.copy()
    df["RoomsPerHousehold"]      = df["AveRooms"]    / df["AveOccup"].replace(0, np.nan)
    df["BedroomRatio"]           = df["AveBedrms"]   / df["AveRooms"].replace(0, np.nan)
    df["PopulationPerHousehold"] = df["Population"]  / df["AveOccup"].replace(0, np.nan)

    # Cap extreme outliers introduced by near-zero denominators
    for col in ["RoomsPerHousehold", "BedroomRatio", "PopulationPerHousehold"]:
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=q99)

    df = df.fillna(df.median(numeric_only=True))
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test."""
    X = df[ENGINEERED_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split: %d train rows / %d test rows", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> Pipeline:
    """
    Return an unfitted sklearn Pipeline that scales all features.
    Fit it on training data only; use .transform() on test / live data.
    """
    return Pipeline([("scaler", StandardScaler())])


def prepare(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, Pipeline]:
    """
    Full preprocessing convenience function.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, fitted_preprocessor
    """
    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled  = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor
