# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — install deps + train the model
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ml/  ml/
COPY models/ models/

# Train and save model artifacts to models/
RUN python -m ml.train --model random_forest --n-estimators 100

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — lean image with only what's needed to serve predictions
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN useradd --no-create-home --no-log-init appuser

# Runtime Python dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Application code
COPY app/  app/
COPY ml/   ml/

# Trained model artifacts from builder stage
COPY --from=builder /app/models/ models/

# Fix ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
