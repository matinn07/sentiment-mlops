from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI()

tokenizer = DistilBertTokenizerFast.from_pretrained("model/")
model = DistilBertForSequenceClassification.from_pretrained("model/")
model.eval()

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    inputs = tokenizer(
        review.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
    label = torch.argmax(outputs.logits).item()
    return {"sentiment": "Positive" if label == 1 else "Negative"}
