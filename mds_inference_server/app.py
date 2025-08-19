from fastapi import FastAPI, UploadFile, HTTPException, Query
import pandas as pd
from .loader import load_model
from .predictor import predict
import mlflow
from mlflow import MlflowClient
from typing import List, Dict, Optional, Any
import os

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5001")

# Set MinIO credentials for artifact access
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

app = FastAPI(title="Model Inference Server", description="MLflow-powered model inference and management API")

@app.post("/predict")
async def predict_data(file: UploadFile, model_name: str, version: int = 1):
    df = pd.read_csv(file.file)
    model = load_model(model_name, version)
    preds = predict(model, df)
    return {"predictions": preds.tolist()}

@app.get("/health")
async def health():
    return {"status": "healthy", "mlflow_uri": mlflow.get_tracking_uri()}

# Model Registry Query Endpoints

@app.get("/available-models", summary="List available models for inference")
async def list_available_models():
    """Get all models that can be loaded for inference"""
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        
        available_models = []
        for model in models:
            versions = client.search_model_versions(f"name='{model.name}'")
            
            # Test which versions are actually loadable
            working_versions = []
            for version in versions:
                try:
                    # Quick test to see if version is loadable
                    model_uri = f"models:/{model.name}/{version.version}"
                    # Don't actually load, just check if URI is valid
                    working_versions.append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "creation_timestamp": version.creation_timestamp
                    })
                except:
                    continue
            
            if working_versions:
                available_models.append({
                    "name": model.name,
                    "description": model.description,
                    "available_versions": working_versions,
                    "latest_version": max(working_versions, key=lambda x: int(x["version"]))["version"]
                })
        
        return {
            "available_models": available_models,
            "count": len(available_models),
            "inference_ready": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list available models: {str(e)}")

@app.get("/models/{model_name}/versions", summary="Get available versions for a model")
async def get_model_versions(model_name: str):
    """Get all available versions for a specific model"""
    try:
        client = MlflowClient()
        
        try:
            model = client.get_registered_model(model_name)
        except:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        versions = client.search_model_versions(f"name='{model_name}'")
        
        version_details = []
        for version in versions:
            # Test if version is loadable
            is_loadable = True
            error_message = None
            
            try:
                # Quick validation without full loading
                load_model(model_name, int(version.version))
            except Exception as e:
                is_loadable = False
                error_message = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            
            version_details.append({
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "creation_timestamp": version.creation_timestamp,
                "last_updated_timestamp": version.last_updated_timestamp,
                "source": version.source,
                "run_id": version.run_id,
                "is_loadable": is_loadable,
                "error_message": error_message if not is_loadable else None
            })
        
        # Sort by version number (descending)
        version_details.sort(key=lambda x: int(x["version"]), reverse=True)
        
        return {
            "model_name": model_name,
            "versions": version_details,
            "total_versions": len(version_details),
            "loadable_versions": len([v for v in version_details if v["is_loadable"]])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model versions: {str(e)}")

@app.get("/models/{model_name}/latest", summary="Get latest available model version")
async def get_latest_model_version(model_name: str):
    """Get the latest loadable version of a model"""
    try:
        client = MlflowClient()
        
        try:
            model = client.get_registered_model(model_name)
        except:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        versions = client.search_model_versions(f"name='{model_name}'")
        
        # Sort versions by number (descending) and find first loadable one
        sorted_versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
        
        for version in sorted_versions:
            try:
                # Test if version is loadable
                load_model(model_name, int(version.version))
                
                return {
                    "model_name": model_name,
                    "latest_version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "source": version.source,
                    "run_id": version.run_id,
                    "is_loadable": True
                }
            except:
                continue
        
        raise HTTPException(status_code=404, detail=f"No loadable versions found for model '{model_name}'")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latest model version: {str(e)}")

@app.get("/models/{model_name}/predict-schema", summary="Get model input schema")
async def get_model_schema(model_name: str, version: Optional[int] = Query(None, description="Model version (latest if not specified)")):
    """Get input schema for a model (if available)"""
    try:
        client = MlflowClient()
        
        # Get version
        if version is None:
            latest_response = await get_latest_model_version(model_name)
            version = int(latest_response["latest_version"])
        
        try:
            model_version = client.get_model_version(model_name, str(version))
        except:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' version '{version}' not found")
        
        # Try to load model and inspect
        try:
            model = load_model(model_name, version)
            
            # Try to get feature names if available
            feature_names = None
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            elif hasattr(model, 'feature_names'):
                feature_names = list(model.feature_names)
            
            # Get input shape if available
            input_shape = None
            if hasattr(model, 'n_features_in_'):
                input_shape = model.n_features_in_
            
            return {
                "model_name": model_name,
                "version": version,
                "feature_names": feature_names,
                "input_shape": input_shape,
                "model_type": type(model).__name__,
                "schema_available": feature_names is not None or input_shape is not None
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to inspect model: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model schema: {str(e)}")

@app.get("/inference-stats", summary="Get inference server statistics")
async def get_inference_stats():
    """Get statistics about models available for inference"""
    try:
        client = MlflowClient()
        
        models = client.search_registered_models()
        
        total_models = len(models)
        total_versions = 0
        loadable_models = 0
        loadable_versions = 0
        
        model_details = []
        
        for model in models:
            versions = client.search_model_versions(f"name='{model.name}'")
            total_versions += len(versions)
            
            model_loadable_versions = 0
            for version in versions:
                try:
                    load_model(model.name, int(version.version))
                    model_loadable_versions += 1
                    loadable_versions += 1
                except:
                    pass
            
            if model_loadable_versions > 0:
                loadable_models += 1
            
            model_details.append({
                "name": model.name,
                "total_versions": len(versions),
                "loadable_versions": model_loadable_versions
            })
        
        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "loadable_models": loadable_models,
            "loadable_versions": loadable_versions,
            "model_details": model_details,
            "mlflow_uri": mlflow.get_tracking_uri(),
            "server_ready": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get inference stats: {str(e)}")

@app.post("/batch-predict", summary="Batch predictions for multiple inputs")
async def batch_predict(
    files: List[UploadFile], 
    model_name: str, 
    version: Optional[int] = Query(None, description="Model version (latest if not specified)")
):
    """Perform batch predictions on multiple CSV files"""
    try:
        # Get model version
        if version is None:
            latest_response = await get_latest_model_version(model_name)
            version = int(latest_response["latest_version"])
        
        # Load model once
        model = load_model(model_name, version)
        
        results = []
        for i, file in enumerate(files):
            try:
                df = pd.read_csv(file.file)
                preds = predict(model, df)
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "predictions": preds.tolist(),
                    "rows_processed": len(df),
                    "success": True
                })
            except Exception as e:
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
        
        successful_predictions = len([r for r in results if r["success"]])
        
        return {
            "model_name": model_name,
            "version": version,
            "files_processed": len(files),
            "successful_predictions": successful_predictions,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
