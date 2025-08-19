import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri("http://localhost:5001")

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

def ensure_experiment():
    """Ensure we have an active experiment"""
    try:
        # Try to create a new experiment with S3 storage to match Docker config
        experiment_name = "model-training"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment with S3 storage
            experiment_id = mlflow.create_experiment(
                experiment_name, 
                artifact_location="s3://mlflow/"
            )
            print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
        
        # Set the active experiment
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"Error managing experiment: {e}")
        # Fallback: create with timestamp
        import time
        fallback_name = f"training-{int(time.time())}"
        experiment_id = mlflow.create_experiment(fallback_name, artifact_location="s3://mlflow/")
        mlflow.set_experiment(fallback_name)
        print(f"Created fallback experiment '{fallback_name}'")
        return experiment_id

def log_model_to_mlflow(model, onnx_model, score, model_name):
    # Ensure we have an active experiment before logging
    ensure_experiment()
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"S3 endpoint: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")
    
    with mlflow.start_run() as run:
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(model, "sklearn-model", registered_model_name=model_name)
        mlflow.onnx.log_model(onnx_model, "onnx-model", registered_model_name=model_name)
        print(f"Logged run with ID: {run.info.run_id}")
        return run.info.run_id