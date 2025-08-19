from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from .trainer import train_models
import traceback

app = FastAPI()

@app.post("/train")
async def train(file: UploadFile):
    try:
        print(f"=== ENDPOINT HIT: Received file: {file.filename} ===")
        
        # Try to import trainer here to see if that's the issue
        print("Attempting to import trainer...")
        from .trainer import train_models
        print("Trainer imported successfully!")
        
        print("Reading CSV file...")
        df = pd.read_csv(file.file)
        print(f"CSV loaded with shape: {df.shape}")
        
        print("Calling train_models...")
        run_ids = train_models(df)
        print(f"Training completed: {run_ids}")
        
        return {"status": "success", "trained_models": run_ids}
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
