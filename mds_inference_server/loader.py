import mlflow.sklearn
import mlflow
import os

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5001")

# Set MinIO credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

def get_latest_model_version(model_name):
    """Get the latest model version that actually has artifacts"""
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        
        # Sort versions by version number (highest first)
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        
        # Try to find a version that actually works
        for version in versions[:3]:  # Check last 3 versions
            try:
                # Test if we can load this version
                mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
                print(f"✅ Found working version: {version.version}")
                return version.version
            except Exception as e:
                print(f"❌ Version {version.version} failed: {str(e)[:100]}...")
                continue
                
        return None
    except Exception as e:
        print(f"Error getting model versions: {e}")
        return None

def load_model(model_name, version=None):
    print(f"Loading model: {model_name}, requested version: {version}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # If no version specified or version fails, try to get latest working version
    if version is None:
        version = get_latest_model_version(model_name)
        if version is None:
            raise Exception(f"No working versions found for model {model_name}")
    
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        print(f"✅ Successfully loaded {model_name} version {version}")
        return model
    except Exception as e:
        print(f"❌ Failed to load version {version}: {e}")
        
        # Try to get latest working version as fallback
        latest_version = get_latest_model_version(model_name)
        if latest_version and latest_version != version:
            print(f"🔄 Trying latest working version: {latest_version}")
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
            print(f"✅ Successfully loaded {model_name} version {latest_version}")
            return model
        
        raise e