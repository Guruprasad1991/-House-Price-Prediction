"""
test_api.py
───────────
Integration tests for the FastAPI endpoints.
Uses httpx's AsyncClient with the app in test mode.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ── Valid sample payload ───────────────────────────────────────────────────────

VALID_HOUSE = {
    "MedInc":     8.3252,
    "HouseAge":   41.0,
    "AveRooms":   6.9812,
    "AveBedrms":  1.0238,
    "Population": 322.0,
    "AveOccup":   2.5556,
    "Latitude":   37.88,
    "Longitude": -122.23,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_predictor():
    """Return a mock Predictor so tests don't need trained artifacts on disk."""
    mock = MagicMock()
    mock.predict_single.return_value = 4.526   # $452,600
    mock.predict_batch.return_value  = [4.526, 3.5]
    mock.model_name = "random_forest"
    mock.metadata   = {
        "model_name":    "random_forest",
        "trained_at":    "2026-01-01T00:00:00+00:00",
        "feature_names": ["MedInc", "HouseAge"],
        "metrics":       {"mae": 0.35, "rmse": 0.52, "r2": 0.82, "mape": 18.5},
    }
    return mock


@pytest.fixture(scope="module")
def client(mock_predictor):
    with patch("app.main.get_predictor", return_value=mock_predictor), \
         patch("app.model.get_predictor", return_value=mock_predictor):
        with TestClient(app) as c:
            yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_fields(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "metrics" in body
        assert "model_name" in body
        assert "trained_at" in body

    def test_metrics_present(self, client):
        metrics = client.get("/health").json()["metrics"]
        for key in ("mae", "rmse", "r2", "mape"):
            assert key in metrics


# ── Single prediction ─────────────────────────────────────────────────────────

class TestPredict:
    def test_valid_request_200(self, client):
        resp = client.post("/predict", json=VALID_HOUSE)
        assert resp.status_code == 200

    def test_response_fields(self, client):
        body = client.post("/predict", json=VALID_HOUSE).json()
        assert "predicted_price_usd" in body
        assert "predicted_price_100k" in body
        assert "model_name" in body

    def test_price_positive(self, client):
        body = client.post("/predict", json=VALID_HOUSE).json()
        assert body["predicted_price_usd"] > 0

    def test_price_usd_is_100k_times_100k(self, client):
        body = client.post("/predict", json=VALID_HOUSE).json()
        assert abs(body["predicted_price_usd"] - body["predicted_price_100k"] * 100_000) < 1

    # ── Validation errors ────────────────────────────────────────────────────

    def test_missing_field_422(self, client):
        payload = {k: v for k, v in VALID_HOUSE.items() if k != "MedInc"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_medinc_422(self, client):
        payload = {**VALID_HOUSE, "MedInc": -1.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_latitude_out_of_range_422(self, client):
        payload = {**VALID_HOUSE, "Latitude": 99.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_bedrooms_exceed_rooms_422(self, client):
        payload = {**VALID_HOUSE, "AveBedrms": 20.0, "AveRooms": 5.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_body_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ── Batch prediction ──────────────────────────────────────────────────────────

class TestBatchPredict:
    def test_batch_two_houses_200(self, client):
        payload = {"houses": [VALID_HOUSE, VALID_HOUSE]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_count_matches(self, client):
        payload = {"houses": [VALID_HOUSE, VALID_HOUSE]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_empty_list_422(self, client):
        resp = client.post("/predict/batch", json={"houses": []})
        assert resp.status_code == 422

    def test_batch_response_structure(self, client):
        payload = {"houses": [VALID_HOUSE]}
        body = client.post("/predict/batch", json=payload).json()
        pred = body["predictions"][0]
        assert "predicted_price_usd" in pred
        assert "model_name" in pred


# ── Root redirect ─────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")
