import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from .onnx_converter import convert_to_onnx
from .mlflow_utils import log_model_to_mlflow

MODEL_REGISTRY = {
    'rf': {
        'constructor': lambda: RandomForestRegressor(n_estimators=50),
        'type': 'rf_regressor'
    },
    'svm': {
        'constructor': lambda: SVR(),
        'type': 'svm_regressor'
    }
}

def train_models(df: pd.DataFrame, model_name: str = 'rf'):
    X = df.drop(columns=["target", "timestamp"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY['rf'])
    model = model_info['constructor']()
    model_type = model_info['type']
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    onnx_model = convert_to_onnx(model, X_train.shape[1])
    run_id = log_model_to_mlflow(model, onnx_model, score, model_type)
    return [run_id]
