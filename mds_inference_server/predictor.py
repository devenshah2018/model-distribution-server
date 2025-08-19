def predict(model, df):
    X = df.drop(columns=["timestamp", "target"], errors="ignore")
    return model.predict(X)
