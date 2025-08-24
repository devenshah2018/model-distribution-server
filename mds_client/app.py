import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from registry import MODEL_REGISTRY

st.set_page_config(page_title="MDS Enterprise Portal", layout="wide", page_icon="📊")

st.title("📊 MDS Enterprise Model Portal")
st.markdown("""
Welcome to the MDS Enterprise Model Portal. Effortlessly upload your data, select a model, and generate predictions with best-in-class machine learning workflows. 

**How it works:**
1. Upload your CSV or select a sample dataset.
2. Preview your data and choose a task type.
3. Select a model, train, and run predictions.
4. View results and a professional summary.
""")

with st.sidebar:
    st.header("Step 1: Data Selection")
    st.write("Choose a sample dataset or upload your own CSV file. Your data is kept secure and private.")
    sample_files = {
        "Time Series Weather": "sample_data/time_series_weather.csv",
        "Classification Data": "sample_data/classification_data.csv"
    }
    sample_choice = st.selectbox("Select a sample or upload your own:", ["Upload your own CSV"] + list(sample_files.keys()), help="Choose a sample to explore or upload your own data for custom predictions.")
    uploaded_file = None
    csv_data = None
    if sample_choice == "Upload your own CSV":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="Accepted format: .csv. Ensure your file is properly formatted.")
        if uploaded_file:
            csv_data = uploaded_file.read().decode("utf-8")
    else:
        with open(sample_files[sample_choice], "r") as f:
            csv_data = f.read()

if csv_data:
    df = pd.read_csv(StringIO(csv_data))
    st.subheader("Step 2: Data Preview")
    st.info("Review your data below. Ensure columns and rows are as expected before proceeding.")
    with st.expander("Show Data Preview", expanded=True):
        st.dataframe(df, use_container_width=True)
        st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("Step 3: Task Type Selection")
    task_type = st.radio(
        "Select the type of prediction you want to perform:",
        ["Regression", "Classification"],
        help="Choose 'Regression' for predicting continuous values, or 'Classification' for categories."
    )

    st.subheader("Step 4: Model Selection & Training")
    st.write("Select a machine learning model best suited for your data and use case.")
    model_reg = MODEL_REGISTRY
    filtered_models = [info['display_name'] for key, info in model_reg.items() if info['category'] == task_type.lower()]
    model_key_map = {info['display_name']: key for key, info in model_reg.items() if info['category'] == task_type.lower()}
    model_name = st.selectbox("Choose Model", filtered_models, help="Pick a model type. Regression for continuous values, classification for categories.")

    # Track training state
    if 'training_success' not in st.session_state:
        st.session_state['training_success'] = False
    version = None
    if not st.session_state['training_success']:
        if st.button("🚀 Train Model", key="train"):
            st.toast("Training model. Please wait...")
            files = {"file": ("input.csv", csv_data)}
            params = {"model_name": model_key_map[model_name]}
            try:
                train_resp = requests.post("http://localhost:8001/train", files=files, data=params)
                train_json = train_resp.json()
                if train_resp.status_code == 200:
                    st.toast(f"Model trained successfully. Ready for prediction.")
                    st.session_state['training_success'] = True
                else:
                    st.toast("Training completed, but no version returned.")
                    st.session_state['training_success'] = False
            except Exception as e:
                st.toast(f"Training failed: {e}")
                st.session_state['training_success'] = False
    if st.session_state['training_success']:
        if st.button("🔮 Run Model Prediction", key="predict") and model_name:
            model_mlflow_name = model_reg[model_key_map[model_name]]['type']
            latest_version = requests.get(f"http://localhost:8002/models/{model_mlflow_name}/latest")
            latest_version_json = latest_version.json()
            version = latest_version_json.get("latest_version", None)
            if latest_version.status_code == 200 and "version" in latest_version_json:
                version = latest_version_json["version"]
            st.toast("Running prediction. This may take a few moments...")
            files = {"file": ("input.csv", csv_data)}
            params = {"model_name": model_name, "version": version}
            try:
                pred_resp = requests.post(f"http://localhost:8002/predict?model_name={model_mlflow_name}&version={version}", files=files)
                pred_json = pred_resp.json()
                predictions = pred_json.get("predictions", [])
                st.subheader("Step 5: Prediction Results")
                st.toast("Predictions generated successfully.")
                with st.expander("Show Predictions", expanded=True):
                    pred_df = pd.DataFrame({"Prediction": predictions})
                    st.dataframe(pred_df, use_container_width=True)
                # Text summary via Llama2
                st.subheader("Step 6: Model & Data Summary")
                exp_id = None
                try:
                    train_models_resp = requests.get(f"http://localhost:8001/models/{model_mlflow_name}")
                    train_json = train_models_resp.json()
                    run_id = train_json.get("versions")[0].get("run_id")
                    run_resp = requests.get(f"http://localhost:8001/runs/{run_id}")
                    run_json = run_resp.json()
                    exp_id = run_json.get("experiment_id", None)
                except Exception:
                    exp_id = None
                if exp_id:
                    try:
                        summary_resp = requests.post(
                            f"http://localhost:8002/summarize?model_name={model_mlflow_name}&experiment_id={exp_id}",
                            files={"data_context": ("input.csv", csv_data)}
                        )
                        summary = summary_resp.json().get("response", "")
                        st.markdown(summary)
                    except Exception:
                        st.warning("Could not fetch summary.")
                else:
                    st.warning("Experiment ID not found for summary.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a CSV or select a sample dataset to begin. For best results, ensure your data is clean and well-formatted.")