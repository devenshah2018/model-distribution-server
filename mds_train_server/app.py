from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from .trainer import train_models
import traceback

app = FastAPI()

@app.post("/train")
async def train(file: UploadFile):
    try:
        from .trainer import train_models        
        df = pd.read_csv(file.file)
        run_ids = train_models(df)
        return {"status": "success", "trained_models": run_ids}
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
