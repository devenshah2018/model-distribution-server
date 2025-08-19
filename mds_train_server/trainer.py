import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from .onnx_converter import convert_to_onnx
from .mlflow_utils import log_model_to_mlflow


def train_models(df: pd.DataFrame):
    # Assume time series dataset: [timestamp, feature1, ..., target]
    X = df.drop(columns=["target", "timestamp"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=50)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    onnx_model = convert_to_onnx(model, X_train.shape[1])
    run_id = log_model_to_mlflow(model, onnx_model, score, "rf_regressor")

    return [run_id]
