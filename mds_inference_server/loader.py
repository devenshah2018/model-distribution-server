import mlflow.sklearn
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5001")

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        for version in versions[:3]:
            try:
                mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
                return version.version
            except Exception as e:
                continue   
        return None
    except Exception as e:
        return None

def load_model(model_name, version=None):
    if version is None:
        version = get_latest_model_version(model_name)
        if version is None:
            raise Exception(f"No working versions found for model {model_name}")
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        return model
    except Exception as e:
        latest_version = get_latest_model_version(model_name)
        if latest_version and latest_version != version:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
            return model
        raise e