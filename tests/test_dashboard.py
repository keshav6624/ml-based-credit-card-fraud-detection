from __future__ import annotations

# IMPORTANT (Windows + TensorFlow import-order quirk): TensorFlow must be imported
# before scikit-learn/joblib in this process, otherwise `_pywrap_tensorflow_internal`
# fails to initialize. app.py and the other test files follow the same convention.
import tensorflow  # noqa: F401  (imported first on purpose)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "artifact_manifest.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app as app_module
from model_loader import (
    FEATURE_NAMES,
    FORBIDDEN_KMEANS_OUTPUT_COLUMNS,
    TARGET_NAME,
    predict,
    predict_batch,
    validate_dataframe,
)


@pytest.fixture(scope="module")
def models():
    from model_loader import load_models

    return load_models(MODEL_DIR)


def _labelled_frame(n: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES)
    )
    frame[TARGET_NAME] = np.array([0, 1, 0])[:n]
    return frame


# ---------------------------------------------------------------------------
# App smoke: pages render without uploaded data / empty states (AppTest)
# ---------------------------------------------------------------------------

def _apptest() -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception
    return at


def test_overview_renders_without_uploaded_data():
    at = _apptest()
    info_values = [info.value for info in at.info]
    assert any("Upload and process a CSV" in value for value in info_values)


def test_single_transaction_page_handles_no_data():
    at = _apptest()
    at.sidebar.radio[0].set_value("Single Transaction Prediction")
    at.run()
    assert not at.exception
    info_values = [info.value for info in at.info]
    assert any("No processed dataset yet" in value for value in info_values)
    assert any("Fraud Detection" in value for value in info_values)


def test_model_health_page_renders_without_uploaded_data():
    at = _apptest()
    at.sidebar.radio[0].set_value("Model Health")
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 3  # artifacts, runtime versions, schema
    warning_values = [warning.value for warning in at.warning]
    assert any("unsupervised" in value for value in warning_values)
    assert any("in-sample" in value for value in warning_values)


# ---------------------------------------------------------------------------
# Manual single transaction: exactly 30 features, Class excluded
# ---------------------------------------------------------------------------

def test_manual_prediction_uses_exactly_30_features_and_excludes_class(models):
    payload = {name: float(index % 7) for index, name in enumerate(FEATURE_NAMES)}
    result = predict(models, payload)

    assert hasattr(models.kmeans, "get") or isinstance(models.kmeans, dict)
    assert set(FORBIDDEN_KMEANS_OUTPUT_COLUMNS).isdisjoint(result)
    assert set(result) == {
        "kmeans_cluster",
        "logistic_regression_class",
        "logistic_regression_probability",
        "neural_network_class",
        "neural_network_probability",
    }
    assert isinstance(result["kmeans_cluster"], int)

    with pytest.raises(ValueError, match="never be passed here"):
        predict(models, {**payload, TARGET_NAME: 1})

    missing_payload = dict(payload)
    missing_payload.pop("Amount")
    with pytest.raises(ValueError, match="missing="):
        predict(models, missing_payload)

    non_finite_payload = dict(payload)
    non_finite_payload["Amount"] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        predict(models, non_finite_payload)

    with pytest.raises(ValueError, match="extra="):
        predict(models, {**payload, "Unlucky": 1.0})


def test_manual_prediction_renders_supervised_fraud_statuses_without_relabeling_kmeans(
    monkeypatch,
):
    messages = {"success": [], "error": [], "info": [], "caption": []}
    for method in messages:
        monkeypatch.setattr(
            app_module.st,
            method,
            lambda message, _method=method, **kwargs: messages[_method].append(message),
        )

    app_module._render_fraud_status(
        {
            "logistic_regression_class": 0,
            "neural_network_class": 1,
            "kmeans_cluster": 0,
        }
    )

    assert messages["success"] == ["Logistic Regression: No Fraud Detected"]
    assert messages["error"] == ["Neural Network: Possible Fraud Detected"]
    assert len(messages["caption"]) == 2
    assert all("not confirmation" in message for message in messages["caption"])
    assert messages["info"] == []


def test_manual_prediction_explains_kmeans_is_not_a_fraud_status(monkeypatch):
    info_messages = []
    monkeypatch.setattr(
        app_module.st,
        "info",
        lambda message, **kwargs: info_messages.append(message),
    )

    app_module._render_fraud_status({"kmeans_cluster": 1})

    assert len(info_messages) == 1
    assert "cluster assignment only" in info_messages[0]
    assert "fraud or non-fraud classification" in info_messages[0]


# ---------------------------------------------------------------------------
# Mode A row selection returns the matching row's input and predictions
# ---------------------------------------------------------------------------

def test_csv_row_selection_returns_matching_row_predictions(models):
    frame = _labelled_frame(3)
    results = predict_batch(models, frame)
    output = app_module.build_output(frame, results)

    for row in range(3):
        view = app_module.transaction_view(frame, output, row)
        original = view["original"]
        pd.testing.assert_series_equal(
            original[list(FEATURE_NAMES)],
            frame.iloc[row][list(FEATURE_NAMES)],
            check_freq=False,
        )
        for column in results.columns:
            assert view["predictions"][column] == results.iloc[row][column]
        assert view["true_label"] == int(frame.iloc[row][TARGET_NAME])

    out_of_range = len(frame)
    with pytest.raises(ValueError, match="Row position must be between"):
        app_module.transaction_view(frame, output, out_of_range)


# ---------------------------------------------------------------------------
# Model Health helpers: manifest handling
# ---------------------------------------------------------------------------

def test_model_health_rows_with_manifest():
    rows = app_module.model_health_rows(MODEL_DIR, MANIFEST_PATH)
    assert len(rows) == 4
    assert all(row["Present"] for row in rows)
    assert all(row["Manifest status"] == "OK (matches manifest)" for row in rows)
    assert all(row["SHA-256"] for row in rows)


def test_model_health_rows_with_missing_manifest(tmp_path):
    rows = app_module.model_health_rows(MODEL_DIR, tmp_path / "does_not_exist.json")
    artifact_rows = [row for row in rows if row["Present"] and row["SHA-256"]]
    assert len(artifact_rows) == 4
    assert all(row["Manifest status"] == "No manifest entry" for row in artifact_rows)
    notes = [row for row in rows if str(row["Artifact"]).startswith("(manifest")]
    assert any("Manifest file not found" in note["Artifact"] for note in notes)


def test_model_health_rows_with_corrupt_manifest(tmp_path):
    bad = tmp_path / "artifact_manifest.json"
    bad.write_text("not-json{", encoding="utf-8")
    rows = app_module.model_health_rows(MODEL_DIR, bad)
    notes = [row for row in rows if str(row["Artifact"]).startswith("(manifest")]
    assert any("could not be read" in note["Artifact"] for note in notes)


def test_runtime_versions_never_crash():
    rows = app_module.runtime_versions()
    components = {row["Component"] for row in rows}
    assert {"Python", "scikit-learn", "TensorFlow"}.issubset(components)
    for row in rows:
        assert isinstance(row["Version"], str)


# ---------------------------------------------------------------------------
# Stale-result invalidation and session reset
# ---------------------------------------------------------------------------

def test_new_upload_invalidates_stale_results():
    state = dict(app_module.DEFAULT_STATE)
    frame = _labelled_frame(2)
    state["input_frame"] = frame
    state["output"] = app_module.build_output(frame, pd.DataFrame({"x": [1, 2]}))
    state["metrics"] = {"logistic_regression": {}}
    state["results_signature"] = "A"
    state["attempt_signature"] = "B"

    assert app_module.is_results_stale(state)
    cleared = app_module.invalidate_stale(state, "B")
    assert cleared
    assert state["output"] is None
    assert state["input_frame"] is None
    assert state["metrics"] == {}
    assert state["metrics_signature"] is None
    assert state["results_signature"] is None
    assert not app_module.is_results_stale(state)


def test_matching_upload_keeps_results():
    state = dict(app_module.DEFAULT_STATE)
    state["results_signature"] = "A"
    state["output"] = object()
    assert not app_module.is_results_stale(state)
    assert not app_module.invalidate_stale(state, "A")
    assert state["output"] is not None


def test_clear_session_restores_defaults():
    state = dict(app_module.DEFAULT_STATE)
    state["input_frame"] = "data"
    state["output"] = "results"
    state["metrics"] = {"logistic_regression": {}}
    state["attempt_signature"] = "zzz"
    state["extra_key"] = "remove me"

    app_module.clear_session(state)
    assert app_module.DEFAULT_STATE.keys() <= state.keys()
    assert state["run_error"] is None
    assert state["input_frame"] is None
    assert state["output"] is None
    assert state["metrics"] == {}
    assert state["attempt_signature"] is None
    assert "extra_key" not in state


# ---------------------------------------------------------------------------
# CSV parsing: BOM, empty, decode, schema rejection
# ---------------------------------------------------------------------------

def _csv_bytes(
    columns: list[str],
    row_count: int = 1,
    extra_columns: list[str] | None = None,
    drop_last: bool = False,
    bom: bool = False,
) -> bytes:
    """Build a synthetic CSV covering `columns`; each row gets one value per column."""
    header = ",".join(list(columns) + list(extra_columns or []))
    rows: list[str] = []
    total = len(columns) + len(extra_columns or [])
    for row_index in range(row_count):
        values = [str(round((index - 12) * 0.37, 4)) for index in range(total)]
        if columns and columns[-1] == TARGET_NAME:
            values[-1] = str(row_index % 2)
        if drop_last:
            values = values[:-1]
        rows.append(",".join(values))
    body = "\n".join(rows)
    text = header + "\n" + body + ("\n" if rows else "\n")
    data = text.encode("utf-8")
    return b"\xef\xbb\xbf" + data if bom else data


def test_parse_upload_bytes_supports_utf8_bom():
    raw = _csv_bytes(list(FEATURE_NAMES) + [TARGET_NAME], row_count=1, bom=True)
    frame = app_module.parse_upload_bytes(raw)
    assert frame.columns[0] == "Time"
    assert frame.columns[-1] == TARGET_NAME
    validation = validate_dataframe(frame)
    assert validation["rows"] == 1 and validation["has_labels"]


def test_parse_upload_bytes_rejects_empty_and_undecodable_content():
    with pytest.raises(ValueError, match="empty"):
        app_module.parse_upload_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        app_module.parse_upload_bytes(None)
    with pytest.raises(ValueError, match="UTF-8"):
        app_module.parse_upload_bytes(b"\xff\xfe. not utf-8")
    header_only = _csv_bytes(list(FEATURE_NAMES), row_count=0)
    with pytest.raises(ValueError, match="contains no data rows"):
        app_module.parse_upload_bytes(header_only)


def test_validation_rejects_unexpected_columns_with_actionable_message():
    raw = _csv_bytes(list(FEATURE_NAMES) + ["ExtraColumn"], row_count=1)
    frame = app_module.parse_upload_bytes(raw)
    with pytest.raises(ValueError, match=r"extra=\[.ExtraColumn.\]"):
        validate_dataframe(frame)
    raw_missing = _csv_bytes(list(FEATURE_NAMES[:-1]), row_count=1)
    frame_missing = app_module.parse_upload_bytes(raw_missing)
    with pytest.raises(ValueError, match=r"missing=\[.Amount.\]"):
        validate_dataframe(frame_missing)
