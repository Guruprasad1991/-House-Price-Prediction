# 🏠 House Price Prediction API

![CI](https://github.com/Guruprasad1991/-House-Price-Prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A **production-ready** machine learning API that predicts California median house values.  
Built with scikit-learn · Served via FastAPI · Containerised with Docker · Auto-deployed via Render.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Guruprasad1991/-House-Price-Prediction)

---

## 🔗 Live Demo

> **API:** `https://house-price-api.onrender.com`  
> **Interactive docs:** `https://house-price-api.onrender.com/docs`

*(The free Render tier sleeps after 15 min of inactivity — the first request after a sleep wakes it in ~30 seconds.)*

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
│       └── ci.yml           # Lint → Test → Docker build on every push
├── app/
│   ├── main.py              # FastAPI routes
│   ├── model.py             # Singleton model loader
│   └── schemas.py           # Pydantic request / response models
├── ml/
│   ├── preprocess.py        # Data loading, feature engineering, scaling
│   ├── train.py             # Training script (CLI)
│   └── evaluate.py          # Metrics & feature-importance report
├── models/                  # Trained artifacts (generated at build time)
├── tests/
│   ├── test_api.py          # FastAPI endpoint tests (17 tests)
│   └── test_model.py        # ML pipeline unit tests (20 tests)
├── Dockerfile               # Multi-stage build
├── docker-compose.yml
├── render.yaml              # Render deployment config
├── requirements.txt
└── requirements-dev.txt
```

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/Guruprasad1991/-House-Price-Prediction.git
cd -House-Price-Prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (saves artifacts to models/)
python -m ml.train

# 5. Start the API
uvicorn app.main:app --reload

# Open: http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t house-price-prediction .
docker run -p 8000:8000 house-price-prediction
```

---

## ☁️ Deploy to Render (free)

Click the button above, or follow these steps:

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New → Web Service**
2. Connect this GitHub repository
3. Render reads `render.yaml` — all settings are pre-configured
4. Click **Deploy** → your API is live in ~3 minutes

Every `git push` to `main` triggers an automatic redeploy.

---

## 🔌 API Reference

### `GET /health`

```bash
curl https://house-price-api.onrender.com/health
```

```json
{
  "status": "ok",
  "model_name": "random_forest",
  "trained_at": "2026-07-02T10:00:00+00:00",
  "metrics": { "mae": 0.332, "rmse": 0.501, "r2": 0.813, "mape": 18.4 },
  "feature_names": ["MedInc", "HouseAge", "AveRooms", "..."],
  "version": "1.0.0"
}
```

### `POST /predict`

```bash
curl -X POST https://house-price-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252, "HouseAge": 41.0,
    "AveRooms": 6.98,  "AveBedrms": 1.02,
    "Population": 322, "AveOccup": 2.56,
    "Latitude": 37.88, "Longitude": -122.23
  }'
```

```json
{
  "predicted_price_usd": 452600.00,
  "predicted_price_100k": 4.526,
  "model_name": "random_forest"
}
```

### `POST /predict/batch`

Send up to 1,000 houses in one request:

```json
{ "houses": [ { ...house1... }, { ...house2... } ] }
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
python -m ml.train --n-estimators 20   # fast train for testing
pytest tests/ -v --cov=app --cov=ml
```

37 tests covering ML pipeline, API validation, error cases, and batch prediction.

---

## 🏋️ Training Options

```bash
python -m ml.train                          # default: Random Forest 100 trees
python -m ml.train --model linear           # Linear Regression (fastest)
python -m ml.train --model ridge            # Ridge Regression
python -m ml.train --n-estimators 300       # larger forest (more accurate)
```

---

## 📊 Model Performance

| Model         | R²    | MAE ($) | RMSE ($) |
|---------------|-------|---------|---------|
| Linear        | 0.607 | 51,200  | 73,800  |
| Ridge         | 0.608 | 51,100  | 73,700  |
| **Random Forest** | **0.813** | **33,200** | **50,100** |

*Evaluated on 20% hold-out of the California Housing dataset (20,640 rows).*

---

## 🔄 CI Pipeline

Every push runs three jobs via GitHub Actions:

| Job | What it checks |
|-----|---------------|
| Lint | black (formatting) + flake8 (style) |
| Test | pytest with coverage report |
| Docker | Build image + smoke-test /health |

---

## 🗺️ Roadmap

- [ ] Streamlit dashboard frontend
- [ ] MLflow experiment tracking
- [ ] Model versioning with DVC
- [ ] Automated retraining on new data

---

## 📄 License

MIT © [Guruprasad1991](https://github.com/Guruprasad1991)
