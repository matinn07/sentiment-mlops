import mlflow
import os
import shutil

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/sentiment-distilbert/Production"
EXPORT_DIR = "model"

# Clean old export
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

os.makedirs(EXPORT_DIR, exist_ok=True)

# Download MLflow model
download_path = mlflow.artifacts.download_artifacts(
    artifact_uri=MODEL_URI,
    dst_path=EXPORT_DIR
)

# 🔥 Flatten artifacts/model → model/
src = os.path.join(download_path, "artifacts", "model")
dst = os.path.join(EXPORT_DIR, "model")

shutil.copytree(src, dst)

print("✅ Model exported to ./model/model (Docker-safe)")
