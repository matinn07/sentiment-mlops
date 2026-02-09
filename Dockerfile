FROM python:3.10-slim

# ===============================
# Environment
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# ===============================
# System deps
# ===============================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Workdir
# ===============================
WORKDIR /app

# ===============================
# Install Python deps
# ===============================
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ===============================
# Copy project
# ===============================
COPY . .

# ===============================
# Expose API port
# ===============================
EXPOSE 8000

# ===============================
# Start FastAPI
# ===============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
