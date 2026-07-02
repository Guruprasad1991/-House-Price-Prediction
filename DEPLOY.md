# Deploy to Render (free)

No credit card required. Your API gets a public HTTPS URL in about 3 minutes.

---

## Option A — One-click deploy button

Click the button in the README. Render reads `render.yaml` and configures everything automatically.

---

## Option B — Manual setup (3 steps)

### Step 1 — Connect your repo

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New → Web Service**
3. Select **Connect a repository** → choose `Guruprasad1991/-House-Price-Prediction`

### Step 2 — Configure the service

Render auto-detects `render.yaml` and fills in these settings for you:

| Setting | Value |
|---------|-------|
| Name | `house-price-api` |
| Environment | Python 3 |
| Build command | `pip install -r requirements.txt && python -m ml.train` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Plan | Free |
| Health check path | `/health` |

### Step 3 — Deploy

Click **Create Web Service**. Render will:
1. Clone your repo
2. Run the build command (installs deps + trains the model)
3. Start the API
4. Show you your live URL: `https://house-price-api.onrender.com`

---

## After deploying

```bash
# Check health
curl https://house-price-api.onrender.com/health

# Make a prediction
curl -X POST https://house-price-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252, "HouseAge": 41.0,
    "AveRooms": 6.98,  "AveBedrms": 1.02,
    "Population": 322, "AveOccup": 2.56,
    "Latitude": 37.88, "Longitude": -122.23
  }'

# Open interactive docs
open https://house-price-api.onrender.com/docs
```

---

## Auto-deploy on push

Once connected, every `git push origin main` triggers a new Render build automatically. No extra config needed.

```bash
# Retrain with better model, push → Render redeploys
python -m ml.train --n-estimators 300
git add models/ && git commit -m "chore: retrain 300 trees"
git push
```

---

## Free tier limits

| Limit | Value |
|-------|-------|
| Requests/month | Unlimited |
| Sleep after inactivity | 15 minutes |
| Wake-up time | ~30 seconds |
| Memory | 512 MB |
| Build time | 400 min/month |

For a portfolio or demo project, the free tier is more than enough.

---

## Updating your live URL in the README

After your first deploy, Render gives you a URL like:
`https://house-price-api.onrender.com`

Update the **Live Demo** section in `README.md` with your actual URL, then push:

```bash
git add README.md
git commit -m "docs: add live Render URL"
git push
```
