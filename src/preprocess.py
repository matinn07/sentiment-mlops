import pandas as pd
import re

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower().strip()

def main():
    df = pd.read_csv("data/IMDB-Dataset.csv")
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    df[["clean_review", "label"]].to_csv("data/processed.csv", index=False)

if __name__ == "__main__":
    main()
