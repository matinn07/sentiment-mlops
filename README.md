# End-to-End MLOps Sentiment Analysis System 🚀

This project demonstrates a **complete, production-grade MLOps pipeline** for **sentiment analysis** using **Transformer-based NLP**, **MLflow**, **FastAPI**, **Drift Detection**, **Retraining**, and **Docker**.

It covers the **entire real-world ML lifecycle**:

> **Train → Track → Register → Deploy → Monitor → Detect Drift → Retrain → Redeploy**

This is **not a toy project** — it mirrors how modern ML systems are built and maintained in production.

---

## 🔍 Problem Statement

Automatically classify text reviews as **Positive** or **Negative**, deploy the model as a scalable API, and continuously **monitor data drift** to ensure long-term model reliability.

---

## 🧠 Model & NLP Approach

- **Model**: DistilBERT (`distilbert-base-uncased`)
- **Task**: Binary sentiment classification
- **Why DistilBERT?**
  - Lightweight Transformer
  - Faster inference than BERT
  - CPU-friendly
  - Widely adopted in production NLP systems

The model is fine-tuned on movie review sentiment data and later adapted to real-world user inputs.

---

## 📊 Model Performance (Tracked via MLflow)

| Metric | Value |
|------|------|
| Accuracy | ~0.83 |
| Precision | ~0.85 |
| Recall | ~0.81 |
| F1-score | ~0.83 |

All metrics are **logged, versioned, and reproducible** using MLflow.

---

## 🧪 Experiment Tracking (MLflow)

Each training run logs:
- Model parameters (epochs, tokenizer length, base model)
- Evaluation metrics (accuracy, precision, recall, F1-score)
- Full model artifacts

This enables:
- Experiment comparison
- Reproducibility
- Data-driven deployment decisions

---

## 🏷️ Model Registry & Versioning

- Models are registered in **MLflow Model Registry**
- Multiple versions are maintained
- A **Production alias** explicitly points to the deployed model

This enables:
- Safe promotion & rollback
- Clear governance
- Seamless deployment without code changes

---

## 🖼️ MLflow UI Screenshots

> Screenshots below are from real experiment runs and registry operations.

### Production Model (Alias-based Promotion)
![Production Model](screenshots/Production_model.png)

### Model Version Details
![Version Details](screenshots/Version1_model.png)

### Training Run Metrics
![Training Metrics](screenshots/Run1.png)

### Experiment Runs Overview
![Runs Overview](screenshots/Run1_overview.png)
![Runs Overview](screenshots/Run2_overview.png)

---

## 🌐 Model Deployment (FastAPI)

The production model is served using **FastAPI** via **MLflow PyFunc**, ensuring consistency between training and inference.

### Endpoint

Response
{
  "timestamp": "2026-02-09T12:34:56.123Z",
  "text": "Movie was amazing",
  "prediction": 1,
  "model_stage": "Production"
}

## Features

### Stateless API
### Version-controlled model loading
### Inference logging for monitoring
### Swagger UI support


Access Swagger UI:

http://127.0.0.1:8000/docs

## 📈 Monitoring & Drift Detection (CORE FEATURE)

- This project implements semantic data drift detection, not just statistical heuristics.

- How Drift Is Detected

- CLS embeddings are extracted using DistilBERT

- Cosine distance is computed between:
Training reference embeddings
Recent inference embeddings

- A rolling window prevents historical dilution

- Drift is triggered only on true domain shift
  
## Example Output
Drift score     : 0.4440
Threshold       : 0.30
Status          : DRIFT

## 🔁 Retraining Pipeline (Closed-Loop MLOps)

When **data drift is detected**, the system follows a controlled retraining workflow:

1. **Recent inference data** (logged from production) is collected
2. The model is **retrained** using this new data
3. A **new model version** is logged and registered in **MLflow**
4. **Reference embeddings** are updated to reflect the new data distribution
5. The retrained model is **manually promoted to Production** via MLflow
6. The **FastAPI service reloads** the new Production model

This completes a **closed-loop MLOps system**, ensuring the model remains reliable as real-world data evolves.

---

## 🐳 Dockerized Deployment

The entire system is **fully containerized** using Docker.

### Why Docker?

- Reproducible environments
- No local dependency conflicts
- Production-aligned deployment
- Easy scaling and rollback

### Build Image
docker build -t sentiment-mlops:latest .

### Run Container
docker run -p 8000:8000 sentiment-mlops:latest

### Or using Docker Compose (Recommended)
docker-compose up --build

### Once running, access the API at:
http://127.0.0.1:8000/docs

## 🧩 Project Structure

sentiment-mlops/
│
├── app/              # FastAPI inference service
├── src/              # Model training pipeline
├── monitoring/       # Drift detection logic
├── retraining/       # Retraining pipeline
├── data/             # Reference vectors (ignored in Git)
├── logs/             # Inference & drift logs (ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── screenshots/      # MLflow UI screenshots
└── README.md

## 🎯 Key MLOps Concepts Demonstrated

- Transformer-based NLP modeling

- Experiment tracking with MLflow

- Model registry & versioning

- Alias-based production deployment

- FastAPI inference serving

- Semantic drift detection (rolling window)

- Retraining pipelines

- Dockerized ML systems

- End-to-end ML lifecycle management

## 👤 Author

### Matin Raheman Nadaf
### B.Tech in Computer Science & Engineering

## Focus Areas:

- MLOps
- NLP
- Production Machine Learning Systems


