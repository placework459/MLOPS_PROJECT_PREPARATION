from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc
from pathlib import Path

# === SET CORRECT FILE URI ===
# === SET CORRECT FILE URI ===
mlruns_path = Path("../training/mlruns").resolve().as_uri()
mlflow.set_tracking_uri(mlruns_path)

# === MODEL PATH ===
MODEL_PATH = 'runs:/e999d4c7e18248f29c76479e8a17bf27/model'

# === LOAD MODEL ===
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise e

# === FASTAPI APP ===
app = FastAPI(title="Churn Prediction API")

class InferenceRequest(BaseModel):
    features: list[dict]

@app.get("/")
def root():
    return {"message": "Churn Prediction API is up."}

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        input_df = pd.DataFrame(request.features)
        if input_df.empty:
            raise ValueError("No input data provided.")
        predictions = model.predict(input_df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
