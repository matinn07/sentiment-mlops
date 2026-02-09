import torch
import mlflow.pyfunc
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

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
