import json
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime
import os

# =============================
# Configuration
# =============================
LOG_FILE = "logs/inference_logs.jsonl"
REF_FILE = "data/reference_vectors.npy"
DRIFT_LOG = "logs/drift.log"

WINDOW_SIZE = 10        # rolling window size
MIN_SAMPLES = 10        # minimum required for drift check
THRESHOLD = 0.30        # cosine distance threshold

# =============================
# Load tokenizer & model
# =============================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)
bert = DistilBertModel.from_pretrained(
    "distilbert-base-uncased"
)
bert.eval()

# =============================
# CLS embedding function
# =============================
def cls_embed(texts, batch_size=8):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = bert(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls.cpu().numpy())

    return np.vstack(embeddings)

# =============================
# Drift detection logic
# =============================
def detect_drift():
    if not os.path.exists(LOG_FILE):
        print("⚠️ No inference logs found")
        return False

    if not os.path.exists(REF_FILE):
        print("❌ Reference vectors missing")
        return False

    # Load texts from inference logs
    texts = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    if len(texts) < MIN_SAMPLES:
        print(f"⚠️ Not enough samples for drift check "
              f"({len(texts)}/{MIN_SAMPLES})")
        return False

    # 🔁 Rolling window
    recent_texts = texts[-WINDOW_SIZE:]

    # Compute embeddings
    current_vec = cls_embed(recent_texts).mean(axis=0).reshape(1, -1)
    reference_vec = np.load(REF_FILE).reshape(1, -1)

    # Cosine distance
    drift_score = cosine_distances(reference_vec, current_vec)[0][0]
    status = "DRIFT" if drift_score > THRESHOLD else "OK"

    # Log drift result
    os.makedirs(os.path.dirname(DRIFT_LOG), exist_ok=True)
    with open(DRIFT_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.utcnow().isoformat()} | "
            f"score={drift_score:.4f} | "
            f"threshold={THRESHOLD} | "
            f"window={WINDOW_SIZE} | "
            f"{status}\n"
        )

    # Console output
    print(">>> DRIFT CHECK <<<")
    print(f"Window size     : {WINDOW_SIZE}")
    print(f"Drift score     : {drift_score:.4f}")
    print(f"Threshold       : {THRESHOLD}")
    print(f"Status          : {status}")

    return drift_score > THRESHOLD

# =============================
# Entry point
# =============================
if __name__ == "__main__":
    if detect_drift():
        print("⚠️ Drift detected – retraining recommended")
    else:
        print("✅ No drift")
