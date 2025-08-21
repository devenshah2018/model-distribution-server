import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    median_absolute_error,
    max_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    root_mean_squared_error,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from .onnx_converter import convert_to_onnx
from .mlflow_utils import log_model_to_mlflow

MODEL_REGISTRY = {
    'rfr': {
        'constructor': lambda: RandomForestRegressor(n_estimators=50),
        'type': 'rf_regressor',
        'category': 'regression'
    },
    'svm': {
        'constructor': lambda: SVR(),
        'type': 'svm_regressor',
        'category': 'regression'
    },
    'rfc': {
        'constructor': lambda: RandomForestClassifier(n_estimators=50),
        'type': 'rf_classifier',
        'category': 'classification'
    }
}

def train_models(df: pd.DataFrame, model_name: str = 'rfr'):
    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[model_name])
    if model_info['category'] == 'regression':
        return train_regression_models(df, model_name)
    if model_info['category'] == 'classification':
        return train_classification_models(df, model_name)

def train_classification_models(df: pd.DataFrame, model_name: str = 'rfc'):
    X = df.drop(columns=["target", "timestamp"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y if len(set(y)) > 1 else None
    )
    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[model_name])
    model = model_info['constructor']()
    model_type = model_info['type']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if len(set(y_test)) < 2:
        print("Warning: Test set contains only one class. Metrics may be unreliable.")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    onnx_model = convert_to_onnx(model, X_train.shape[1])
    run_id = log_model_to_mlflow(model, onnx_model, metrics, model_type)
    return [run_id]

def train_regression_models(df: pd.DataFrame, model_name: str = 'rfr'):
    X = df.drop(columns=["target", "timestamp"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[model_name])
    model = model_info['constructor']()
    model_type = model_info['type']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "explained_variance": explained_variance_score(y_test, y_pred),
        "median_absolute_error": median_absolute_error(y_test, y_pred),
        "max_error": max_error(y_test, y_pred),
        "mean_squared_log_error": mean_squared_log_error(y_test, y_pred),
        "mean_poisson_deviance": mean_poisson_deviance(y_test, y_pred),
        "mean_gamma_deviance": mean_gamma_deviance(y_test, y_pred),
        "mean_tweedie_deviance": mean_tweedie_deviance(y_test, y_pred, power=1.5),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, y_pred),
        "mean_pinball_loss": mean_pinball_loss(y_test, y_pred, alpha=0.5)
    }
    onnx_model = convert_to_onnx(model, X_train.shape[1])
    run_id = log_model_to_mlflow(model, onnx_model, metrics, model_type)
    return [run_id]
