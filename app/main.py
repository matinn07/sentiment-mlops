from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import json
from datetime import datetime
import os
import pandas as pd

# =============================
# App setup
# =============================
app = FastAPI(title="Sentiment Analysis API")

# =============================
# Logging setup
# =============================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "inference_logs.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# =============================
# MLflow model loading
# =============================
MODEL_URI = "models:/sentiment-distilbert/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)

# =============================
# Request schema
# =============================
class InputText(BaseModel):
    text: str

# =============================
# Response schema (NEW)
# =============================
class PredictionResponse(BaseModel):
    timestamp: str
    text: str
    prediction: int
    model_stage: str

# =============================
# Health check
# =============================
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "sentiment-distilbert",
        "stage": "Production"
    }

# =============================
# Prediction endpoint
# =============================
@app.post("/predict", response_model=PredictionResponse)
def predict(input: InputText):
    df = pd.DataFrame({"text": [input.text]})
    prediction = model.predict(df)
    pred_label = int(prediction[0])

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": input.text,
        "prediction": pred_label,
        "model_stage": "Production"
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry
