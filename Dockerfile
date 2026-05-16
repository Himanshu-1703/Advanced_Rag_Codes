FROM python:3.12

WORKDIR /app

# ── Layer 1: install deps (cached until requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: backend package ───────────────────────────────────────────────────
COPY backend/ backend/

# ── Layer 3: documents ─────────────────────────────────────────────────────────
COPY documents/ documents/

# ── Layer 4: application files (change most often — last for cache efficiency) ─
COPY app.py .
COPY evaluate.py .
COPY main.py .
COPY goldens.json .
COPY sessions.json .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
