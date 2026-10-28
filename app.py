import hashlib
import io
import json
import os
import platform

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow  # noqa: F401  (imported before sklearn/joblib on Windows)
import pandas as pd
import streamlit as st

from model_loader import (
    FEATURE_NAMES,
    MODEL_DISPLAY_NAMES,
    MODEL_DIR,
    TARGET_NAME,
    evaluate_predictions,
    load_models,
    predict,
    predict_batch,
    validate_dataframe,
)

PROJECT_ROOT = MODEL_DIR.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "artifact_manifest.json"

ARTIFACT_FILES = (
    "kmeans.joblib",
    "logistic_regression.joblib",
    "neural_network.keras",
    "neural_network_preprocessing.joblib",
)
_SUPERVISED_CLASS_COLUMNS = ("logistic_regression_class", "neural_network_class")
_RESULT_COLUMNS = (
    "logistic_regression_class",
    "logistic_regression_probability",
    "neural_network_class",
    "neural_network_probability",
    "kmeans_cluster",
)
_RESULT_LABELS = {
    "logistic_regression_probability": "Logistic Regression score",
    "logistic_regression_class": "Logistic Regression class",
    "neural_network_probability": "Neural Network score",
    "neural_network_class": "Neural Network class",
    "kmeans_cluster": "K-Means cluster ID",
}
NAV_PAGES = [
    "Overview",
    "Fraud Detection",
    "Model Comparison",
    "Transaction Explorer",
    "Single Transaction Prediction",
    "Model Health",
]
DEFAULT_STATE = {
    "input_frame": None,
    "results": None,
    "output": None,
    "metrics": {},
    "metrics_signature": None,
    "selected_models": ("logistic_regression", "neural_network", "kmeans"),
    "validation": None,
    "validation_error": None,
    "pending_frame": None,
    "run_error": None,
    "upload_signature": None,
    "attempt_signature": None,
    "results_signature": None,
}
TEMPLATE_COLUMNS = list(FEATURE_NAMES)
TEMPLATE_WITH_CLASS_COLUMNS = list(FEATURE_NAMES) + [TARGET_NAME]


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit calls; importable by pytest)
# ---------------------------------------------------------------------------

def file_signature(raw: bytes) -> str:
    """Stable identifier of an uploaded file's exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def parse_upload_bytes(raw: bytes | None) -> pd.DataFrame:
    """Read an uploaded CSV as UTF-8 (BOM tolerant) with actionable errors."""
    if raw is None or len(raw) == 0:
        raise ValueError("Uploaded file is empty. Choose a CSV with the 30 feature columns.")
    try:
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Uploaded file is empty or contains no CSV rows.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Uploaded CSV must be UTF-8 encoded (with or without a BOM); "
            f"text decoding failed ({exc})"
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed on a malformed row: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError("Uploaded CSV is empty or contains no data rows.")
    return frame


def invalidate_stale(state: dict, signature: str) -> bool:
    """Drop predictions/metrics produced for a different upload. Returns True if cleared."""
    stale = state.get("results_signature") is not None and state["results_signature"] != signature
    if stale:
        state["results"] = None
        state["output"] = None
        state["input_frame"] = None
        state["metrics"] = {}
        state["metrics_signature"] = None
        state["results_signature"] = None
    return stale


def is_results_stale(state: dict) -> bool:
    """True when the latest upload attempt does not match the processed results."""
    attempt = state.get("attempt_signature")
    results_signature = state.get("results_signature")
    return results_signature is not None and attempt not in (None, results_signature)


def clear_session(state: dict) -> None:
    """Clear all session values (including widgets), then restore safe defaults."""
    for key in list(state.keys()):
        try:
            del state[key]
        except (KeyError, TypeError):
            continue
    for key, value in DEFAULT_STATE.items():
        state[key] = value


def build_output(input_frame: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Combine validated input and prediction results, preserving row order."""
    output = input_frame.copy()
    output.insert(0, "Source row", range(len(output)))
    return pd.concat(
        [output.reset_index(drop=True), results.reset_index(drop=True)], axis=1
    )


def transaction_view(
    input_frame: pd.DataFrame, output: pd.DataFrame, row_position: int
) -> dict:
    """Original values, available predictions, and optional ground truth for one row."""
    if input_frame is None or output is None:
        raise ValueError("No processed dataset is available for this view")
    if not 0 <= row_position < len(input_frame):
        raise ValueError(
            f"Row position must be between 0 and {len(input_frame) - 1}; got {row_position}"
        )
    original = input_frame.iloc[row_position]
    output_row = output.iloc[row_position]
    predictions = {}
    for column in _RESULT_COLUMNS:
        if column in output.columns:
            value = output_row[column]
            predictions[column] = int(value) if column.endswith(("class", "cluster")) else float(value)
    true_label = None
    if TARGET_NAME in output.columns:
        true_label = int(output_row[TARGET_NAME])
    return {
        "original": original,
        "predictions": predictions,
        "true_label": true_label,
        "row_position": row_position,
    }


def model_health_rows(models_dir, manifest_path) -> list[dict]:
    """Artifact inventory with sizes, hashes, and manifest comparison (never raises)."""
    expected_hashes: dict[str, str] = {}
    manifest_note = None
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
        if not isinstance(entries, list):
            manifest_note = "Manifest exists but does not list artifacts"
        else:
            for entry in entries:
                if isinstance(entry, dict) and entry.get("path") and entry.get("sha256"):
                    expected_hashes[os.path.basename(str(entry["path"]))] = str(entry["sha256"]).lower()
    except FileNotFoundError:
        manifest_note = "Manifest file not found (hashes cannot be cross-checked)"
    except (OSError, ValueError) as exc:
        manifest_note = f"Manifest could not be read ({type(exc).__name__})"

    rows = []
    for name in ARTIFACT_FILES:
        path = os.path.join(models_dir, name)
        row = {"Artifact": name, "Present": False, "Size (bytes)": None, "SHA-256": None, "Manifest status": "â€”"}
        if not os.path.exists(path):
            row["Manifest status"] = "File missing"
            rows.append(row)
            continue
        row["Present"] = True
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
            row["Size (bytes)"] = len(raw)
            row["SHA-256"] = hashlib.sha256(raw).hexdigest()
        except OSError as exc:
            row["Manifest status"] = f"File exists but could not be read ({type(exc).__name__})"
            rows.append(row)
            continue
        expected = expected_hashes.get(name)
        if expected is None:
            row["Manifest status"] = "No manifest entry"
        elif expected != row["SHA-256"]:
            row["Manifest status"] = "Hash mismatch vs manifest"
        else:
            row["Manifest status"] = "OK (matches manifest)"
        rows.append(row)
    if manifest_note and rows:
        rows.append({"Artifact": f"(manifest: {manifest_note})", "Present": True, "Size (bytes)": None,
                     "SHA-256": None, "Manifest status": "Note"})
    return rows


def runtime_versions() -> list[dict]:
    """Best-effort runtime version table; missing metadata shows as unavailable."""
    rows = [{"Component": "Python", "Version": platform.python_version()}]
    for component, importer in (
        ("scikit-learn", lambda: __import__("sklearn").__version__),
        ("TensorFlow", lambda: __import__("tensorflow").__version__),
        ("Keras", lambda: __import__("keras").__version__),
        ("joblib", lambda: __import__("joblib").__version__),
        ("pandas", lambda: __import__("pandas").__version__),
        ("numpy", lambda: __import__("numpy").__version__),
        ("Streamlit", lambda: __import__("streamlit").__version__),
    ):
        try:
            rows.append({"Component": component, "Version": str(importer())})
        except Exception:  # noqa: BLE001 â€” metadata must never break the page
            rows.append({"Component": component, "Version": "unavailable"})
    return rows


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

@st.cache_resource
def get_models():
    return load_models()


def _metric_card(label: str, value: int | float | None, default_text: str = "0") -> None:
    if value is None:
        st.metric(label, default_text)
    elif isinstance(value, bool):
        st.metric(label, str(int(value)))
    elif isinstance(value, int):
        st.metric(label, f"{value:,}" if abs(value) >= 1000 else str(value))
    else:
        st.metric(label, f"{value:.4f}")


def _show_stale_banner() -> None:
    if is_results_stale(st.session_state):
        st.warning(
            "The predictions currently shown were produced from a previously processed "
            "dataset. Run predictions again on the latest upload to refresh them."
        )


def _prediction_cards(predictions: dict[str, float | int]) -> None:
    if not predictions:
        st.info("No predictions were produced for this transaction (select models on the Fraud Detection page).")
        return
    cards = st.columns(max(len(predictions), 1))
    for card, (column, value) in zip(cards, predictions.items()):
        with card:
            label = _RESULT_LABELS.get(column, column)
            if column.endswith("cluster"):
                st.metric(label, f"Cluster {value}")
            elif isinstance(value, int):
                st.metric(label, str(value))
            else:
                st.metric(label, f"{value:.4f}")
    if "kmeans_cluster" in predictions:
        st.caption("K-Means output is a cluster ID only â€” unsupervised and never a fraud label.")


def _render_overview() -> None:
    output = st.session_state.output
    input_frame = st.session_state.input_frame
    if output is None or input_frame is None:
        st.markdown("## Overview")
        st.info("Upload and process a CSV from Fraud Detection to populate the overview.")
        return

    _show_stale_banner()
    st.markdown("## Overview")
    total = len(output)
    supervised_columns = [column for column in _SUPERVISED_CLASS_COLUMNS if column in output]
    primary = supervised_columns[0] if supervised_columns else None
    fraud_count = int(output[primary].sum()) if primary else None
    cards = st.columns(4)
    with cards[0]:
        _metric_card("Transactions processed", total)
    with cards[1]:
        _metric_card("Predicted fraud", fraud_count)
    with cards[2]:
        _metric_card("Predicted legitimate", total - fraud_count if fraud_count is not None else None)
    with cards[3]:
        _metric_card("Actual fraud", int(input_frame[TARGET_NAME].sum()) if TARGET_NAME in input_frame else 0)

    left, right = st.columns(2)
    with left:
        st.subheader("Actual class distribution")
        if TARGET_NAME in input_frame:
            st.bar_chart(input_frame[TARGET_NAME].value_counts().sort_index())
        else:
            st.info("Actual labels were not supplied.")
    with right:
        st.subheader("Predicted class distribution")
        if primary:
            st.bar_chart(output[primary].value_counts().sort_index())
        else:
            st.info("Select at least one supervised model to show predicted classes.")

    st.subheader("Model status")
    status = pd.DataFrame(
        [
            {"Model": "Logistic Regression", "Status": "Ready" if "logistic_regression_class" in output else "Not run", "Output": "Class + probability"},
            {"Model": "Neural Network", "Status": "Ready" if "neural_network_class" in output else "Not run", "Output": "Class + probability"},
            {"Model": "K-Means", "Status": "Ready" if "kmeans_cluster" in output else "Not run", "Output": "Cluster ID only; saved mapping ignored"},
        ]
    )
    st.dataframe(status, hide_index=True, width="stretch")


def _render_detection(models) -> None:
    st.markdown("## Fraud Detection")
    st.caption("Upload the original creditcard.csv format (UTF-8, BOM tolerated). Predictions run only after validation and confirmation.")
    uploaded = st.file_uploader("Upload transaction CSV", type="csv")
    if uploaded is not None:
        signature = file_signature(uploaded.getvalue())
        st.session_state.attempt_signature = signature
        try:
            candidate = parse_upload_bytes(uploaded.getvalue())
            validation = validate_dataframe(candidate)
        except (TypeError, ValueError, OSError) as exc:
            st.session_state.validation = None
            st.session_state.validation_error = str(exc)
            st.session_state.pending_frame = None
        else:
            st.session_state.validation = validation
            st.session_state.validation_error = None
            st.session_state.pending_frame = candidate
            st.session_state.upload_signature = signature
            invalidate_stale(st.session_state, signature)

    if st.session_state.validation_error:
        st.error("Dataset validation failed â€” fix the CSV and re-upload.")
        st.error(str(st.session_state.validation_error))
        if st.session_state.output is not None:
            _show_stale_banner()
            st.caption("Existing results from the previous dataset stay available until a new validation succeeds.")
        return
    if st.session_state.validation:
        validation = st.session_state.validation
        st.success(
            f"Valid dataset: {validation['rows']:,} rows and 30 ordered numeric features"
            + (" with binary Class labels." if validation["has_labels"] else ".")
        )
        pending = st.session_state.pending_frame
        st.dataframe(pending.head(10), width="stretch")
        choices = list(MODEL_DISPLAY_NAMES)
        selected_labels = st.multiselect(
            "Models to run",
            choices,
            default=list(st.session_state.selected_models),
            format_func=lambda key: MODEL_DISPLAY_NAMES[key],
        )
        selected = tuple(selected_labels)
        st.session_state.selected_models = selected
        if st.button("Run predictions", type="primary", disabled=not selected):
            with st.spinner("Running selected models..."):
                try:
                    results = predict_batch(models, pending, selected)
                except (TypeError, ValueError, OSError, RuntimeError) as exc:
                    st.session_state.run_error = f"{type(exc).__name__}: {exc}"
                    st.error("Prediction failed â€” no new results were saved.")
                    st.error(st.session_state.run_error)
                else:
                    st.session_state.run_error = None
                    output = build_output(pending, results)
                    st.session_state.input_frame = pending
                    st.session_state.results = results
                    st.session_state.output = output
                    st.session_state.results_signature = st.session_state.upload_signature
                    metrics = evaluate_predictions(pending, results) if TARGET_NAME in pending else {}
                    st.session_state.metrics = metrics
                    st.session_state.metrics_signature = (
                        (st.session_state.upload_signature, selected) if metrics else None
                    )
                    st.success("Predictions completed. Use the other dashboard pages to explore results.")
        with st.expander("Download CSV templates"):
            st.caption("Header-only templates. Feature template contains the 30 required columns; the labelled template adds the optional Class column.")
            st.download_button(
                "Download feature template (30 columns)",
                pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False),
                "fraud_features_template.csv",
                "text/csv",
                key="template_features_download",
            )
            st.download_button(
                "Download labelled template (30 columns + Class)",
                pd.DataFrame(columns=TEMPLATE_WITH_CLASS_COLUMNS).to_csv(index=False),
                "fraud_features_class_template.csv",
                "text/csv",
                key="template_class_download",
            )
        if st.session_state.run_error:
            st.error(f"Last prediction attempt failed: {st.session_state.run_error}")
        if st.session_state.output is not None:
            _show_stale_banner()
            st.subheader("Latest prediction results")
            st.dataframe(st.session_state.output.head(500), width="stretch")
            st.download_button(
                "Download prediction results",
                st.session_state.output.to_csv(index=False),
                "fraud_predictions.csv",
                "text/csv",
                key="results_download",
            )


def _metrics_table_signature_caption() -> str | None:
    signature = st.session_state.metrics_signature
    if not signature:
        return None
    dataset_signature, selection = signature
    models_text = ", ".join(MODEL_DISPLAY_NAMES.get(key, key) for key in selection)
    if st.session_state.upload_signature != dataset_signature:
        return "Metrics below were computed on a previous dataset."
    if tuple(st.session_state.selected_models) != tuple(selection):
        return f"Current model selection differs; these metrics were computed for: {models_text}."
    return f"Metrics were computed for: {models_text}."


def _render_comparison() -> None:
    st.markdown("## Model Comparison")
    metrics = st.session_state.metrics
    if not metrics:
        st.info(
            "Process a labelled CSV with at least one supervised model selected to "
            "calculate in-sample evaluation. Each supervised model is scored independently."
        )
        return
    _show_stale_banner()
    st.warning("These are in-sample evaluation results on the uploaded data, not generalization performance.")
    caption_note = _metrics_table_signature_caption()
    if caption_note:
        st.caption(caption_note)
    frame_rows = []
    for model, values in metrics.items():
        frame_rows.append(
            {
                "Model": MODEL_DISPLAY_NAMES[model],
                **{key: values[key] for key in ("precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc")},
            }
        )
    table = pd.DataFrame(frame_rows).set_index("Model")
    st.dataframe(table, width="stretch")
    chart_columns = [column for column in ("precision", "recall", "f1", "balanced_accuracy") if column in table.columns]
    st.bar_chart(table[chart_columns])
    for model, values in metrics.items():
        st.subheader(f"{MODEL_DISPLAY_NAMES[model]} confusion matrix")
        st.dataframe(
            pd.DataFrame(values["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]),
            width="stretch",
        )
    st.caption("K-Means is unsupervised (cluster IDs only) and is intentionally excluded from supervised metrics.")


def _render_explorer() -> None:
    st.markdown("## Transaction Explorer")
    output = st.session_state.output
    if output is None:
        st.info("Process a CSV from Fraud Detection first.")
        return
    _show_stale_banner()
    filtered = output
    search = st.text_input("Search source row or feature values")
    if search:
        mask = filtered.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False))
        filtered = filtered[mask.any(axis=1)]
    model_options = [key for key in ("logistic_regression", "neural_network") if f"{key}_class" in output]
    selected_model = st.selectbox("Filter by model", ["All"] + model_options, format_func=lambda key: MODEL_DISPLAY_NAMES.get(key, key))
    if selected_model != "All":
        predicted = st.selectbox("Predicted class", [0, 1])
        filtered = filtered[filtered[f"{selected_model}_class"] == predicted]
    st.caption(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} matching rows.")
    st.dataframe(filtered.head(500), width="stretch")
    st.download_button("Download filtered results", filtered.to_csv(index=False), "filtered_predictions.csv", "text/csv", key="filtered_download")


def _render_manual(models) -> None:
    st.caption(
        "V1â€“V28 are PCA-transformed features from the credit-card dataset â€” enter genuine "
        "feature values from a real transaction row rather than inventing real-world "
        "interpretations for these codes."
    )
    lead = st.columns(2)
    with lead[0]:
        st.number_input("Time", value=0.0, format="%.6f", key="manual_Time")
    with lead[1]:
        st.number_input("Amount", value=0.0, format="%.6f", key="manual_Amount")
    st.caption("Transformed features (V1â€“V28)")
    grid = st.columns(4)
    for index, name in enumerate(FEATURE_NAMES[1:29]):
        with grid[index % 4]:
            st.number_input(name, value=0.0, format="%.6f", key=f"manual_{name}")
    if st.button("Reset manual inputs to defaults", key="manual_reset"):
        for name in FEATURE_NAMES:
            st.session_state[f"manual_{name}"] = 0.0
        st.rerun()
    payload = {name: float(st.session_state[f"manual_{name}"]) for name in FEATURE_NAMES}
    if st.button("Score transaction", type="primary", key="manual_score"):
        try:
            result = predict(models, payload)
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.success("Transaction scored with the loaded artifacts.")
            _prediction_cards(result)
            st.caption(
                "Probabilities are model scores, not calibrated real-world fraud "
                "probabilities. The logistic and neural-network classes use their default "
                "0.5 decision rules."
            )


def _render_single(models) -> None:
    st.markdown("## Single Transaction Prediction")
    st.caption(
        "Score one transaction with the loaded artifacts. Probabilities are model scores, "
        "not calibrated real-world fraud probabilities."
    )
    input_frame = st.session_state.input_frame
    output = st.session_state.output
    mode = st.radio("Input mode", ["Select a processed row", "Manual feature entry"], horizontal=True)
    if mode == "Select a processed row":
        if input_frame is None or output is None:
            st.info(
                "No processed dataset yet. Open the Fraud Detection page, upload and "
                "validate a CSV, then run predictions â€” this mode reuses that dataset."
            )
            return
        _show_stale_banner()
        max_row = len(input_frame) - 1
        row_position = st.number_input(
            "Row position (0-based)", min_value=0, max_value=max_row, value=0, step=1
        )
        view = transaction_view(input_frame, output, int(row_position))
        left, right = st.columns(2)
        with left:
            st.subheader(f"Original input values (row {row_position})")
            st.dataframe(
                pd.DataFrame(
                    [view["original"][list(FEATURE_NAMES)].to_numpy()],
                    columns=list(FEATURE_NAMES),
                ),
                hide_index=True,
                width="stretch",
            )
        with right:
            st.subheader("Predictions for this transaction")
            _prediction_cards(view["predictions"])
            if view["true_label"] is not None:
                st.metric("Ground-truth Class (from uploaded labels)", str(view["true_label"]))
        st.caption("Ground-truth labels come from the uploaded data and are never sent to the models.")
    else:
        _render_manual(models)


def _render_model_health(models) -> None:
    st.markdown("## Model Health")
    st.success("Model artifacts loaded successfully; dashboard predictions are backed by them.")
    rows = model_health_rows(MODEL_DIR, MANIFEST_PATH)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    mismatches = [row["Artifact"] for row in rows if row["Manifest status"] == "Hash mismatch vs manifest"]
    if mismatches:
        st.error(f"Artifact hash mismatch vs manifest: {', '.join(mismatches)}. Investigation required before trusting predictions.")
    else:
        missing_entries = [row["Artifact"] for row in rows if row["Manifest status"] in ("No manifest entry", "File missing", "File exists but could not be read")]
        if missing_entries:
            st.warning(f"Manifest coverage gaps or unreadable files: {', '.join(missing_entries)}.")
    st.subheader("Runtime versions")
    st.dataframe(pd.DataFrame(runtime_versions()), hide_index=True, width="stretch")

    st.subheader("Expected input schema")
    st.dataframe(
        pd.DataFrame({"Feature order": list(FEATURE_NAMES)}).T,
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"{len(FEATURE_NAMES)} required features (Time, V1â€“V28, Amount) plus an optional "
        f"{TARGET_NAME} binary label column; extras and non-finite values are rejected."
    )

    st.subheader("Decision thresholds")
    neural_threshold = models.neural_network_preprocessing.get("threshold")
    cards = st.columns(2)
    with cards[0]:
        _metric_card("Neural Network threshold", float(neural_threshold) if isinstance(neural_threshold, (int, float)) else None, "unavailable")
    with cards[1]:
        _metric_card("Logistic Regression threshold", 0.5)
    st.caption(
        "The Logistic Regression threshold is its default 0.5 decision rule "
        "(sklearn predict behaviour); it is not a fitted or calibrated threshold."
    )

    st.warning(
        "K-Means cluster IDs are unsupervised. The saved cluster-to-class mapping is "
        "unvalidated and is therefore ignored â€” cluster IDs must never be treated as "
        "fraud labels."
    )
    st.warning(
        "Metrics calculated from uploaded labels may be in-sample and do not establish "
        "generalization performance."
    )


def _init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    st.set_page_config(
        page_title="Fraud Detection Command Center",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #0b1220; }
        [data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid #253552; }
        [data-testid="stMetric"] { background: #111b2e; border: 1px solid #253552;
          border-radius: 10px; padding: 14px; }
        .hero { padding: 10px 0 18px; }
        .hero h1 { margin-bottom: 4px; color: #e6edf7; }
        .hero p { color: #9fb0c8; }
        .status { padding: 12px 16px; border-radius: 8px; background: #12243a;
          border: 1px solid #2d4d70; color: #c9d8ea; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _init_state()
    st.markdown(
        "<div class='hero'><h1>Fraud Detection Command Center</h1>"
        "<p>Artifact-backed transaction monitoring with transparent model outputs.</p></div>",
        unsafe_allow_html=True,
    )
    try:
        models = get_models()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not load the saved models: {exc}")
        st.stop()

    page = st.sidebar.radio("Navigation", NAV_PAGES)
    st.sidebar.caption("K-Means clusters are not fraud labels. Its saved mapping is unvalidated.")
    with st.sidebar.expander("Reset / Clear session"):
        st.caption("Clears the uploaded dataset, predictions, metrics, and manual inputs.")
        confirm = st.checkbox("I understand this clears session data.", key="reset_confirm")
        if st.button("Reset session", disabled=not confirm, key="reset_button"):
            clear_session(st.session_state)
            st.rerun()

    if page == "Overview":
        _render_overview()
    elif page == "Fraud Detection":
        _render_detection(models)
    elif page == "Model Comparison":
        _render_comparison()
    elif page == "Transaction Explorer":
        _render_explorer()
    elif page == "Single Transaction Prediction":
        _render_single(models)
    else:
        _render_model_health(models)


if __name__ == "__main__":
    main()
