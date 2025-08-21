from fastapi import FastAPI, UploadFile, HTTPException, Query, Form
import pandas as pd
from .trainer import train_models
import traceback
import mlflow
from mlflow import MlflowClient
from typing import List, Dict, Optional, Any
import os
import boto3

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5001")

# Set MinIO credentials for artifact access
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

app = FastAPI(title="Model Training Server", description="MLflow-powered model training and management API")

@app.post("/train")
async def train(file: UploadFile, model_name: str = Form('rfr')):
    try:
        from .trainer import train_models        
        df = pd.read_csv(file.file)
        run_ids = train_models(df, model_name=model_name)
        return {"status": "success", "trained_models": run_ids}
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "mlflow_uri": mlflow.get_tracking_uri()}

# MLflow Query Endpoints

@app.get("/experiments", summary="List all experiments")
async def list_experiments():
    """Get all experiments from MLflow"""
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        result = []
        for exp in experiments:
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": exp.creation_time,
                "last_update_time": exp.last_update_time,
                "tags": dict(exp.tags) if exp.tags else {}
            })
        
        return {"experiments": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")

@app.get("/experiments/{experiment_id}/runs", summary="Get runs for an experiment")
async def list_runs(
    experiment_id: str,
    max_results: Optional[int] = Query(100, description="Maximum number of runs to return")
):
    """Get all runs for a specific experiment"""
    try:
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results
        )
        
        result = []
        for run in runs:
            result.append({
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": dict(run.data.metrics) if run.data.metrics else {},
                "params": dict(run.data.params) if run.data.params else {},
                "tags": dict(run.data.tags) if run.data.tags else {}
            })
        
        return {"runs": result, "count": len(result), "experiment_id": experiment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")

@app.get("/models", summary="List all registered models")
async def list_models():
    """Get all registered models from MLflow Model Registry"""
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        
        result = []
        for model in models:
            # Get latest versions for this model
            latest_versions = client.get_latest_versions(model.name)
            
            versions_info = []
            for version in latest_versions:
                versions_info.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "source": version.source,
                    "run_id": version.run_id
                })
            
            result.append({
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "tags": dict(model.tags) if model.tags else {},
                "latest_versions": versions_info
            })
        
        return {"models": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/models/{model_name}", summary="Get model details")
async def get_model_details(model_name: str):
    """Get detailed information about a specific model"""
    try:
        client = MlflowClient()
        
        # Get model info
        try:
            model = client.get_registered_model(model_name)
        except:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        versions_info = []
        for version in versions:
            versions_info.append({
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "creation_timestamp": version.creation_timestamp,
                "last_updated_timestamp": version.last_updated_timestamp,
                "description": version.description,
                "source": version.source,
                "run_id": version.run_id,
                "tags": dict(version.tags) if version.tags else {}
            })
        
        return {
            "name": model.name,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "description": model.description,
            "tags": dict(model.tags) if model.tags else {},
            "versions": versions_info,
            "total_versions": len(versions_info)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

@app.get("/models/{model_name}/versions/{version}", summary="Get specific model version details")
async def get_model_version_details(model_name: str, version: str):
    """Get detailed information about a specific model version"""
    try:
        client = MlflowClient()
        
        try:
            model_version = client.get_model_version(model_name, version)
        except:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' version '{version}' not found")
        
        # Get run details if available
        run_info = None
        if model_version.run_id:
            try:
                run = client.get_run(model_version.run_id)
                run_info = {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics) if run.data.metrics else {},
                    "params": dict(run.data.params) if run.data.params else {}
                }
            except:
                pass
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "status": model_version.status,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "description": model_version.description,
            "source": model_version.source,
            "run_id": model_version.run_id,
            "tags": dict(model_version.tags) if model_version.tags else {},
            "run_info": run_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model version details: {str(e)}")

@app.get("/runs/{run_id}", summary="Get run details")
async def get_run_details(run_id: str):
    """Get detailed information about a specific run"""
    try:
        client = MlflowClient()
        
        try:
            run = client.get_run(run_id)
        except:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        
        # Get artifacts
        artifacts = []
        try:
            artifact_list = client.list_artifacts(run_id)
            for artifact in artifact_list:
                artifacts.append({
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size
                })
        except:
            pass
        
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "lifecycle_stage": run.info.lifecycle_stage,
            "user_id": run.info.user_id,
            "metrics": dict(run.data.metrics) if run.data.metrics else {},
            "params": dict(run.data.params) if run.data.params else {},
            "tags": dict(run.data.tags) if run.data.tags else {},
            "artifacts": artifacts
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run details: {str(e)}")

@app.get("/metrics", summary="Get metrics across all runs")
async def get_metrics_summary(
    experiment_id: Optional[str] = Query(None, description="Filter by experiment ID"),
    metric_name: Optional[str] = Query(None, description="Filter by metric name")
):
    """Get a summary of metrics across runs"""
    try:
        client = MlflowClient()
        
        # Get experiments to search
        experiment_ids = [experiment_id] if experiment_id else [exp.experiment_id for exp in client.search_experiments()]
        
        all_runs = []
        for exp_id in experiment_ids:
            runs = client.search_runs(experiment_ids=[exp_id], max_results=1000)
            all_runs.extend(runs)
        
        metrics_summary = {}
        
        for run in all_runs:
            if run.data.metrics:
                for metric, value in run.data.metrics.items():
                    if metric_name and metric != metric_name:
                        continue
                        
                    if metric not in metrics_summary:
                        metrics_summary[metric] = {
                            "values": [],
                            "runs": [],
                            "min": None,
                            "max": None,
                            "avg": None,
                            "count": 0
                        }
                    
                    metrics_summary[metric]["values"].append(value)
                    metrics_summary[metric]["runs"].append({
                        "run_id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "value": value
                    })
        
        # Calculate statistics
        for metric, data in metrics_summary.items():
            values = data["values"]
            if values:
                data["min"] = min(values)
                data["max"] = max(values)
                data["avg"] = sum(values) / len(values)
                data["count"] = len(values)
                # Remove raw values to reduce response size
                del data["values"]
        
        return {
            "metrics": metrics_summary,
            "total_runs": len(all_runs),
            "filtered_experiments": experiment_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics summary: {str(e)}")
    
@app.get("/stats", summary="Get MLflow statistics")
async def get_mlflow_stats():
    """Get overall statistics about MLflow data"""
    try:
        client = MlflowClient()
        
        # Get experiment count
        experiments = client.search_experiments()
        active_experiments = [exp for exp in experiments if exp.lifecycle_stage == "active"]
        
        # Get model count
        models = client.search_registered_models()
        
        # Get total runs count (approximate)
        total_runs = 0
        for exp in active_experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
            # This is an approximation - MLflow doesn't provide direct count
            exp_runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=10000)
            total_runs += len(exp_runs)
        
        # Get model versions count
        total_versions = 0
        for model in models:
            versions = client.search_model_versions(f"name='{model.name}'")
            total_versions += len(versions)
        
        return {
            "experiments": {
                "total": len(experiments),
                "active": len(active_experiments),
                "deleted": len(experiments) - len(active_experiments)
            },
            "models": {
                "total": len(models),
                "total_versions": total_versions
            },
            "runs": {
                "total": total_runs
            },
            "mlflow_uri": mlflow.get_tracking_uri()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get MLflow stats: {str(e)}")

@app.post("/purge", summary="Delete all MLflow experiments, models, and MinIO bucket contents")
async def purge():
    try:
        client = MlflowClient()

        # Delete all experiments
        experiments = client.search_experiments()
        for exp in experiments:
            try:
                client.delete_experiment(exp.experiment_id)
            except Exception as e:
                pass  # Ignore errors for already deleted experiments

        # Delete all registered models
        models = client.search_registered_models()
        for model in models:
            try:
                client.delete_registered_model(model.name)
            except Exception as e:
                pass  # Ignore errors for already deleted models

        # Clear all objects in the MLflow bucket on MinIO
        s3_endpoint = os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID', 'minio')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', 'minio123')
        bucket_name = 'mlflow'  # Default MLflow bucket name

        s3 = boto3.resource(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        bucket = s3.Bucket(bucket_name)
        bucket.objects.all().delete()

        return {"status": "success", "message": "All MLflow experiments, models, and MinIO bucket contents deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear MLflow and MinIO: {str(e)}")
