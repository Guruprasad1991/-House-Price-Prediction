# 🏠 House Price Prediction API

![CI](https://github.com/Guruprasad1991/-House-Price-Prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A **production-ready** machine learning service that predicts California median house values.  
Built with scikit-learn, served via FastAPI, containerised with Docker, and tested with pytest.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLIENT                           │
│         curl / browser / any HTTP client            │
└────────────────────┬────────────────────────────────┘
                     │  HTTP
┌────────────────────▼────────────────────────────────┐
│              FastAPI  (app/main.py)                 │
│  GET /health   POST /predict   POST /predict/batch  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│           Predictor singleton (app/model.py)        │
│   loads model.joblib + preprocessor.joblib once     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│            sklearn Pipeline (ml/)                   │
│  preprocess.py → train.py → evaluate.py             │
│  Artifacts saved to  models/                        │
└─────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
house-price-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions: lint → test → docker build
├── app/
│   ├── main.py              # FastAPI routes
│   ├── model.py             # Singleton model loader
│   └── schemas.py           # Pydantic request / response models
├── ml/
│   ├── preprocess.py        # Data loading, feature engineering, scaling
│   ├── train.py             # Training script (CLI)
│   └── evaluate.py          # Metrics & feature-importance report
├── models/                  # Trained artifacts (gitignored)
│   ├── model.joblib
│   ├── preprocessor.joblib
│   └── metadata.json
├── tests/
│   ├── test_api.py          # FastAPI endpoint tests
│   └── test_model.py        # ML pipeline unit tests
├── Dockerfile               # Multi-stage build
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/Guruprasad1991/-House-Price-Prediction.git
cd -House-Price-Prediction

# 2. Build image (trains model inside Docker)
docker build -t house-price-prediction .

# 3. Run
docker run -p 8000:8000 house-price-prediction

# 4. Open the interactive docs
open http://localhost:8000/docs
```

Or with docker-compose (after training locally first):

```bash
python -m ml.train          # train & save artifacts to models/
docker-compose up --build
```

---

### Option B — Local (virtualenv)

```bash
# 1. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python -m ml.train

# 4. Start the API
uvicorn app.main:app --reload

# 5. Open docs
open http://localhost:8000/docs
```

---

## 🔌 API Reference

### `GET /health`
Returns model status and training metrics.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_name": "random_forest",
  "trained_at": "2026-07-02T10:00:00+00:00",
  "metrics": {
    "mae": 0.3321,
    "rmse": 0.5012,
    "r2": 0.8134,
    "mape": 18.42
  },
  "feature_names": ["MedInc", "HouseAge", "..."],
  "version": "1.0.0"
}
```

---

### `POST /predict`
Predict the median house value for a single house.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc":     8.3252,
    "HouseAge":   41.0,
    "AveRooms":   6.98,
    "AveBedrms":  1.02,
    "Population": 322.0,
    "AveOccup":   2.56,
    "Latitude":   37.88,
    "Longitude": -122.23
  }'
```

```json
{
  "predicted_price_usd": 452600.00,
  "predicted_price_100k": 4.526,
  "model_name": "random_forest",
  "confidence_note": "Point estimate only. See /health for model metrics."
}
```

---

### `POST /predict/batch`
Predict for up to 1,000 houses in one call.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "houses": [
      {"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
       "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23},
      {"MedInc": 3.5, "HouseAge": 25, "AveRooms": 5.0, "AveBedrms": 1.1,
       "Population": 800, "AveOccup": 3.0, "Latitude": 34.05, "Longitude": -118.24}
    ]
  }'
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Train model first (needed for integration tests)
python -m ml.train --n-estimators 20

# Run all tests with coverage
pytest tests/ -v --cov=app --cov=ml

# Run only fast unit tests (no training)
pytest tests/test_api.py -v
```

---

## 🏋️ Training Options

```bash
# Default: Random Forest (best accuracy)
python -m ml.train

# Linear Regression (fastest, interpretable)
python -m ml.train --model linear

# Ridge Regression (linear + regularisation)
python -m ml.train --model ridge --alpha 0.5

# Larger Random Forest (higher accuracy, slower)
python -m ml.train --model random_forest --n-estimators 300
```

---

## 📊 Model Performance

| Model         | R²    | MAE ($) | RMSE ($) |
|---------------|-------|---------|---------|
| Linear        | 0.607 | 51,200  | 73,800  |
| Ridge         | 0.608 | 51,100  | 73,700  |
| Random Forest | 0.813 | 33,200  | 50,100  |

*Evaluated on 20% hold-out test set of the California Housing dataset.*

---

## 🔄 CI/CD Pipeline

Every push to `main` or `develop` triggers:

1. **Lint** — black (format) + flake8 (style)
2. **Test** — pytest with coverage report
3. **Docker** — build image + smoke-test `/health` endpoint

See `.github/workflows/ci.yml` for details.

---

## 🗺️ Roadmap

- [ ] MLflow experiment tracking
- [ ] Streamlit dashboard frontend
- [ ] Model versioning with DVC
- [ ] Deploy to AWS/GCP/Azure
- [ ] Automated retraining on new data

---

## 📄 License

MIT © [Guruprasad1991](https://github.com/Guruprasad1991)
