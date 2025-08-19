import mlflow
import mlflow.sklearn

def log_model_to_mlflow(model, onnx_model, score, model_name):
    with mlflow.start_run() as run:
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(model, "sklearn-model", registered_model_name=model_name)
        mlflow.onnx.log_model(onnx_model, "onnx-model", registered_model_name=model_name)
        return run.info.run_id
