import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("model/")
model = DistilBertForSequenceClassification.from_pretrained("model/")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    label = torch.argmax(outputs.logits).item()
    return "Positive" if label == 1 else "Negative"

print(predict("I loved the acting and story"))
