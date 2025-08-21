import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri("http://localhost:5001")

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

def ensure_experiment():
    try:
        experiment_name = "model-training"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name, 
                artifact_location="s3://mlflow/"
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        import time
        fallback_name = f"training-{int(time.time())}"
        experiment_id = mlflow.create_experiment(fallback_name, artifact_location="s3://mlflow/")
        mlflow.set_experiment(fallback_name)
        return experiment_id

def log_model_to_mlflow(model, onnx_model, metrics, model_name):
    ensure_experiment()   
    with mlflow.start_run() as run:
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "sklearn-model", registered_model_name=model_name)
        mlflow.onnx.log_model(onnx_model, "onnx-model", registered_model_name=model_name)
        return run.info.run_id