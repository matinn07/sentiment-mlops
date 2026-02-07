import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/processed.csv")

# Optional speed-up for CPU
df = df.sample(20000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# -------------------------
# Tokenizer
# -------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_enc = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=64
)

test_enc = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=64
)

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDBDataset(train_enc, y_train)
test_dataset = IMDBDataset(test_enc, y_test)

# -------------------------
# Model
# -------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# -------------------------
# Metrics function
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -------------------------
# Training arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    report_to="none"
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -------------------------
# Train & evaluate
# -------------------------
trainer.train()
metrics = trainer.evaluate()

# -------------------------
# MLflow logging
# -------------------------
with mlflow.start_run():
    mlflow.log_param("model", "distilbert-base-uncased")
    mlflow.log_param("epochs", 1)
    mlflow.log_param("max_length", 64)

    mlflow.log_metric("accuracy", metrics["eval_accuracy"])
    mlflow.log_metric("precision", metrics["eval_precision"])
    mlflow.log_metric("recall", metrics["eval_recall"])
    mlflow.log_metric("f1", metrics["eval_f1"])

    mlflow.pytorch.log_model(model, "sentiment_model")

# -------------------------
# Save model
# -------------------------
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")
