import os
import json
import numpy as np
import torch
import mlflow
import mlflow.pyfunc
import pandas as pd

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =============================
# MLflow setup
# =============================
MODEL_NAME = "sentiment-distilbert"
mlflow.set_experiment("sentiment-retraining")

# =============================
# Load inference data
# =============================
texts, labels = [], []

with open("logs/inference_logs.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        texts.append(d["text"])
        labels.append(d["prediction"])

labels = pd.Series(labels)

# =============================
# Tokenizer
# =============================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64
)

# =============================
# Dataset
# =============================
class DriftDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

dataset = DriftDataset(encodings, labels)

# =============================
# Model
# =============================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to("cpu")

# =============================
# Training arguments
# =============================
training_args = TrainingArguments(
    output_dir="./retrain_results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    report_to="none",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# =============================
# Retrain
# =============================
trainer.train()

# =============================
# Save model locally (REQUIRED)
# =============================
os.makedirs("model", exist_ok=True)
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")

# =============================
# Update reference vectors
# =============================
def get_cls_embeddings(texts, tokenizer, model, batch_size=16):
    model.eval()
    all_emb = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model.distilbert(**enc)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_emb.append(cls_emb.cpu().numpy())

    return np.vstack(all_emb)

os.makedirs("data", exist_ok=True)
np.save(
    "data/reference_vectors.npy",
    get_cls_embeddings(texts, tokenizer, model).mean(axis=0)
)

# =============================
# MLflow PyFunc wrapper (SAME AS train.py)
# =============================
class SentimentPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            context.artifacts["model_dir"]
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            context.artifacts["model_dir"]
        )
        self.model.eval()

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**enc)
            preds = torch.argmax(outputs.logits, dim=1)

        return preds.numpy()

# =============================
# MLflow logging (CORRECT)
# =============================
with mlflow.start_run():
    mlflow.log_param("retraining_samples", len(texts))

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SentimentPyFunc(),
        artifacts={"model_dir": "model"},
        registered_model_name=MODEL_NAME
    )

print("✅ Retraining complete. New model version registered in MLflow.")
print("➡ Promote the new version to alias: production")
