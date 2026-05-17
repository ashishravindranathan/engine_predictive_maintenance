import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

CONFIG_FILE = Path(__file__).resolve().parent / "deployment_config.json"
CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

HF_REPO_ID = CONFIG["HF_REPO_ID"]
HF_DATA_PATH = CONFIG["HF_DATA_PATH"]
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", CONFIG["HF_MODEL_REPO_ID"])
MODEL_FILE_NAME = CONFIG["MODEL_FILE_NAME"]
FEATURE_COLUMNS_FILE_NAME = CONFIG["FEATURE_COLUMNS_FILE_NAME"]
LABEL_MAPPING_FILE_NAME = CONFIG["LABEL_MAPPING_FILE_NAME"]

INPUT_LOG_FILE = Path("/tmp/inference_inputs.csv")

st.set_page_config(
    page_title="Engine Predictive Maintenance",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=MODEL_FILE_NAME,
        repo_type="model"
    )
    feature_columns_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=FEATURE_COLUMNS_FILE_NAME,
        repo_type="model"
    )
    label_mapping_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=LABEL_MAPPING_FILE_NAME,
        repo_type="model"
    )

    model = joblib.load(model_path)
    feature_columns = json.loads(Path(feature_columns_path).read_text(encoding="utf-8"))
    label_mapping = json.loads(Path(label_mapping_path).read_text(encoding="utf-8"))
    return model, feature_columns, label_mapping

def save_inputs(input_df: pd.DataFrame) -> None:
    write_header = not INPUT_LOG_FILE.exists()
    input_df.to_csv(
        INPUT_LOG_FILE,
        mode="a",
        header=write_header,
        index=False
    )

# Default input values used to (re)initialize the form
DEFAULT_INPUTS = {
    "Engine rpm": 800,
    "Lub oil pressure": 3.5,
    "Fuel pressure": 6.0,
    "Coolant pressure": 2.0,
    "lub oil temp": 75.0,
    "Coolant temp": 85.0,
}

# Widget keys (one per input) so we can reset them via session_state
INPUT_KEYS = {name: f"input_{name}" for name in DEFAULT_INPUTS}

def init_session_state():
    for name, default_value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(INPUT_KEYS[name], default_value)
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("batch_results", None)

def reset_form():
    # Restore every input widget to its default value and clear the last result.
    # Called via on_click BEFORE widgets are re-instantiated on rerun.
    for name, default_value in DEFAULT_INPUTS.items():
        st.session_state[INPUT_KEYS[name]] = default_value
    st.session_state["last_prediction"] = None

def build_input_dataframe(feature_columns):
    user_inputs = {
        "Engine rpm": st.number_input(
            "Engine rpm", min_value=0, step=1, key=INPUT_KEYS["Engine rpm"]
        ),
        "Lub oil pressure": st.number_input(
            "Lub oil pressure", min_value=0.0, step=0.1, format="%.4f",
            key=INPUT_KEYS["Lub oil pressure"]
        ),
        "Fuel pressure": st.number_input(
            "Fuel pressure", min_value=0.0, step=0.1, format="%.4f",
            key=INPUT_KEYS["Fuel pressure"]
        ),
        "Coolant pressure": st.number_input(
            "Coolant pressure", min_value=0.0, step=0.1, format="%.4f",
            key=INPUT_KEYS["Coolant pressure"]
        ),
        "lub oil temp": st.number_input(
            "lub oil temp", min_value=0.0, step=0.1, format="%.4f",
            key=INPUT_KEYS["lub oil temp"]
        ),
        "Coolant temp": st.number_input(
            "Coolant temp", min_value=0.0, step=0.1, format="%.4f",
            key=INPUT_KEYS["Coolant temp"]
        ),
    }

    input_df = pd.DataFrame([user_inputs])
    input_df = input_df[feature_columns]
    return input_df

st.title("Engine Predictive Maintenance")
st.caption("Model is loaded from the Hugging Face Model Hub.")

st.sidebar.subheader("Source Dataset")
st.sidebar.write(f"Repository: {HF_REPO_ID}")
st.sidebar.write(f"File: {HF_DATA_PATH}")

try:
    model, feature_columns, label_mapping = load_artifacts()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.stop()

# Initialize session state defaults BEFORE any widget is created
init_session_state()

def run_batch_predictions(df, model, feature_columns, label_mapping):
    # Validate columns
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns: " + ", ".join(missing)
        )

    # Keep only feature columns in the right order, coerce to numeric
    features_df = df[feature_columns].copy()
    for col in feature_columns:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    invalid_mask = features_df.isna().any(axis=1)
    valid_df = features_df.loc[~invalid_mask]

    predictions = model.predict(valid_df).astype(int)
    prediction_labels = [label_mapping.get(str(p), str(p)) for p in predictions]

    fault_probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
            positive_index = classes.index(1) if 1 in classes else -1
            fault_probabilities = model.predict_proba(valid_df)[:, positive_index]
        except Exception:
            fault_probabilities = None

    # Build a result frame aligned to the ORIGINAL row order (NaN for invalid rows)
    result_df = df.copy()
    result_df["Prediction"] = pd.NA
    result_df["Prediction Label"] = pd.NA
    if fault_probabilities is not None:
        result_df["Fault Probability"] = pd.NA

    result_df.loc[~invalid_mask, "Prediction"] = predictions
    result_df.loc[~invalid_mask, "Prediction Label"] = prediction_labels
    if fault_probabilities is not None:
        result_df.loc[~invalid_mask, "Fault Probability"] = fault_probabilities

    return result_df, int(invalid_mask.sum())

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab_single:
    with st.form("prediction_form"):
        input_df = build_input_dataframe(feature_columns)
        submitted = st.form_submit_button("Predict Engine Condition")

    if submitted:
        save_inputs(input_df)

        prediction = int(model.predict(input_df)[0])
        prediction_label = label_mapping.get(str(prediction), str(prediction))

        fault_probability = None
        if hasattr(model, "predict_proba"):
            try:
                classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
                positive_index = classes.index(1) if 1 in classes else -1
                fault_probability = float(model.predict_proba(input_df)[0][positive_index])
            except Exception:
                fault_probability = None

        # Persist result so it survives reruns (e.g. after a reset click)
        st.session_state["last_prediction"] = {
            "input_df": input_df,
            "prediction_label": prediction_label,
            "fault_probability": fault_probability,
        }

    # Render the most recent prediction (if any) and a Reset button to clear inputs
    last_prediction = st.session_state.get("last_prediction")
    if last_prediction is not None:
        st.subheader("Input DataFrame")
        st.dataframe(last_prediction["input_df"], use_container_width=True)

        st.subheader("Prediction")
        st.success(f"Predicted Engine Condition: {last_prediction['prediction_label']}")

        if last_prediction["fault_probability"] is not None:
            st.metric("Fault Probability", f"{last_prediction['fault_probability']:.2%}")

        st.info(f"Inputs saved to: {INPUT_LOG_FILE}")

        # Button to clear the form back to default values for the next prediction.
        # on_click runs BEFORE the next rerun, so widget defaults are restored cleanly.
        st.button(
            "Reset / New Prediction",
            on_click=reset_form,
            help="Clear inputs and prediction so you can run another prediction without refreshing the page."
        )

with tab_batch:
    st.markdown(
        "Upload a CSV file containing sensor readings for one or more engines "
        "(for example, a daily or periodic export). The app will return one prediction per row."
    )
    st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in feature_columns))

    # Offer a template CSV so users know the exact format
    template_df = pd.DataFrame([{c: DEFAULT_INPUTS.get(c, 0) for c in feature_columns}])
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="engine_input_template.csv",
        mime="text/csv",
        help="Download a single-row template with the correct column headers."
    )

    uploaded_file = st.file_uploader(
        "Upload sensor readings CSV",
        type=["csv"],
        key="batch_csv_uploader"
    )

    col_predict, col_clear = st.columns([1, 1])
    with col_predict:
        run_batch = st.button("Run Batch Predictions", disabled=uploaded_file is None)
    with col_clear:
        clear_batch = st.button("Clear batch results")

    if clear_batch:
        st.session_state["batch_results"] = None

    if run_batch and uploaded_file is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            batch_input_df = None

        if batch_input_df is not None:
            try:
                results_df, dropped_rows = run_batch_predictions(
                    batch_input_df, model, feature_columns, label_mapping
                )
                st.session_state["batch_results"] = {
                    "results_df": results_df,
                    "dropped_rows": dropped_rows,
                    "source_name": uploaded_file.name,
                }
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

    batch_results = st.session_state.get("batch_results")
    if batch_results is not None:
        results_df = batch_results["results_df"]
        st.subheader(f"Predictions for: {batch_results['source_name']}")

        total = len(results_df)
        flagged = int((results_df["Prediction"] == 1).sum())
        normal = int((results_df["Prediction"] == 0).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Total rows", total)
        m2.metric("Predicted Faulty", flagged)
        m3.metric("Predicted Normal", normal)

        if batch_results["dropped_rows"]:
            st.warning(
                f"{batch_results['dropped_rows']} row(s) had missing or non-numeric "
                "values and could not be scored (their Prediction is blank)."
            )

        st.dataframe(results_df, use_container_width=True)

        st.download_button(
            "Download predictions as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{batch_results['source_name']}",
            mime="text/csv"
        )
