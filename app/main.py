from fastapi import FastAPI
from pydantic import BaseModel
import json
from datetime import datetime
import os
import pandas as pd
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# =============================
# App setup
# =============================
app = FastAPI(title="Sentiment Analysis API")

BASE_DIR = "/app"
MODEL_DIR = os.path.join(BASE_DIR, "model", "artifacts", "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "inference_logs.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

tokenizer = None
model = None

# =============================
# Load model on startup
# =============================
@app.on_event("startup")
def load_model():
    global tokenizer, model

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model.eval()
    print("✅ Hugging Face model loaded directly")

# =============================
# Schemas
# =============================
class InputText(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    timestamp: str
    text: str
    prediction: int

# =============================
# Health
# =============================
@app.get("/")
def health():
    return {"status": "ok", "model_source": "huggingface-local"}

# =============================
# Predict
# =============================
@app.post("/predict", response_model=PredictionResponse)
def predict(input: InputText):
    inputs = tokenizer(
        [input.text],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = int(torch.argmax(outputs.logits, dim=1).item())

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": input.text,
        "prediction": pred
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry
