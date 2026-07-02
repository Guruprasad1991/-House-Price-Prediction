"""
test_model.py
─────────────
Unit tests for the ML pipeline: preprocessing, feature engineering,
training, and prediction correctness.
"""

import numpy as np
import pandas as pd
import pytest

from ml.preprocess import (
    ENGINEERED_FEATURES,
    RAW_FEATURES,
    engineer_features,
    load_data,
    prepare,
    split_data,
)
from ml.evaluate import regression_report
from ml.train import get_model, train


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_data()


@pytest.fixture(scope="module")
def engineered_df(raw_df):
    return engineer_features(raw_df)


@pytest.fixture(scope="module")
def prepared_data():
    return prepare(test_size=0.2, random_state=42)


# ── Data loading ──────────────────────────────────────────────────────────────

class TestLoadData:
    def test_shape(self, raw_df):
        assert raw_df.shape[0] > 20_000
        assert raw_df.shape[1] == 9  # 8 features + target

    def test_no_nulls(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_target_positive(self, raw_df):
        assert (raw_df["MedHouseVal"] > 0).all()

    def test_raw_columns_present(self, raw_df):
        for col in RAW_FEATURES:
            assert col in raw_df.columns, f"Missing column: {col}"


# ── Feature engineering ───────────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_new_columns_added(self, engineered_df):
        assert "RoomsPerHousehold" in engineered_df.columns
        assert "BedroomRatio" in engineered_df.columns
        assert "PopulationPerHousehold" in engineered_df.columns

    def test_no_nulls_after_engineering(self, engineered_df):
        assert engineered_df[ENGINEERED_FEATURES].isnull().sum().sum() == 0

    def test_bedroom_ratio_in_range(self, engineered_df):
        # BedroomRatio should be between 0 and 1
        assert (engineered_df["BedroomRatio"] >= 0).all()
        assert (engineered_df["BedroomRatio"] <= 1.5).all()  # allow small outliers after clip

    def test_rooms_per_household_positive(self, engineered_df):
        assert (engineered_df["RoomsPerHousehold"] > 0).all()

    def test_original_columns_unchanged(self, raw_df, engineered_df):
        for col in RAW_FEATURES:
            assert col in engineered_df.columns


# ── Preprocessing pipeline ────────────────────────────────────────────────────

class TestPrepare:
    def test_output_shapes(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        total = len(y_train) + len(y_test)
        assert abs(len(y_test) / total - 0.2) < 0.01  # ≈20% test

    def test_scaled_mean_near_zero(self, prepared_data):
        X_train, _, _, _, _ = prepared_data
        col_means = np.mean(X_train, axis=0)
        assert np.allclose(col_means, 0, atol=0.1)

    def test_scaled_std_near_one(self, prepared_data):
        X_train, _, _, _, _ = prepared_data
        col_stds = np.std(X_train, axis=0)
        assert np.allclose(col_stds, 1, atol=0.1)

    def test_feature_count(self, prepared_data):
        X_train, _, _, _, _ = prepared_data
        assert X_train.shape[1] == len(ENGINEERED_FEATURES)


# ── Model selection ───────────────────────────────────────────────────────────

class TestGetModel:
    def test_linear(self):
        m = get_model("linear")
        assert hasattr(m, "fit")

    def test_ridge(self):
        m = get_model("ridge", alpha=0.5)
        assert m.alpha == 0.5

    def test_random_forest(self):
        m = get_model("random_forest", n_estimators=10)
        assert m.n_estimators == 10

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("xgboost")


# ── Training quality ─────────────────────────────────────────────────────────

class TestTraining:
    """
    Integration test: train a small random forest and check metric thresholds.
    Uses a small n_estimators so CI stays fast.
    """

    @pytest.fixture(scope="class")
    def trained_metrics(self):
        return train(model_name="random_forest", n_estimators=20, test_size=0.2)

    def test_r2_above_threshold(self, trained_metrics):
        assert trained_metrics["r2"] >= 0.70, (
            f"R² too low: {trained_metrics['r2']:.4f}"
        )

    def test_mae_reasonable(self, trained_metrics):
        # MAE should be < $100k (i.e. < 1.0 in $100k units)
        assert trained_metrics["mae"] < 1.0

    def test_mape_reasonable(self, trained_metrics):
        assert trained_metrics["mape"] < 30.0


# ── Evaluation utilities ──────────────────────────────────────────────────────

class TestRegressionReport:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = regression_report(y, y, label="Test")
        assert metrics["mae"]  == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"]   == pytest.approx(1.0)

    def test_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full_like(y_true, y_true.mean())
        metrics = regression_report(y_true, y_pred, label="Test")
        assert metrics["r2"] == pytest.approx(0.0, abs=1e-6)
