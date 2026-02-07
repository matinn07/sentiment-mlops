FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for sklearn, torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache layer)
COPY requirements.txt .

# Install CPU-only torch first (important)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app/ app/
COPY model/ model/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
