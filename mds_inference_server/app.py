from fastapi import FastAPI, UploadFile
import pandas as pd
from .loader import load_model
from .predictor import predict

app = FastAPI()

@app.post("/predict")
async def predict_data(file: UploadFile, model_name: str, version: int = 1):
    df = pd.read_csv(file.file)
    model = load_model(model_name, version)
    preds = predict(model, df)
    return {"predictions": preds.tolist()}
