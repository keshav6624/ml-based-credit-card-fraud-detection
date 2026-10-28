from __future__ import annotations

# IMPORTANT (Windows + TensorFlow import-order quirk): TensorFlow must be imported
# before scikit-learn/joblib in this environment, otherwise
# `_pywrap_tensorflow_internal` fails to initialize ("DLL load failed ... initialization
# routine failed") when sklearn is already loaded in the process. app.py imports
# TensorFlow first for the same reason.
import tensorflow  # noqa: F401  (imported first on purpose)

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_loader import (
    DEFAULT_CHUNK_SIZE,
    FEATURE_NAMES,
    FORBIDDEN_KMEANS_OUTPUT_COLUMNS,
    TARGET_NAME,
    evaluate_predictions,
    load_models,
    predict,
    predict_batch,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def models():
    return load_models(MODEL_DIR)


@pytest.fixture(scope="module")
def features():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(size=(16, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES)
    )


def _labelled_values() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {**{name: 0.0 for name in FEATURE_NAMES}, TARGET_NAME: 0},
            {**{name: 0.0 for name in FEATURE_NAMES}, TARGET_NAME: 1},
        ]
    )


# --------------------------------------------------------------------------
# Synthetic artifact builders for malformed-artifact tests (never touch models/)
# --------------------------------------------------------------------------

def _fitted_scaler(columns=None) -> StandardScaler:
    columns = list(columns) if columns is not None else list(FEATURE_NAMES)
    return StandardScaler().fit(pd.DataFrame(np.zeros((4, len(columns))), columns=columns))


def _valid_kmeans_artifact(**overrides) -> dict:
    x = pd.DataFrame(
        np.repeat(np.array([[0.0], [1.0]]), len(FEATURE_NAMES), axis=1),
        columns=list(FEATURE_NAMES),
    )
    model = KMeans(n_clusters=2, n_init=1, random_state=0).fit(x)
    artifact = {
        "scaler": _fitted_scaler(),
        "model": model,
        "cluster_to_class": {0: 0, 1: 0},
        "feature_names": list(FEATURE_NAMES),
    }
    artifact.update(overrides)
    return artifact


def _valid_logistic_artifact(**overrides) -> dict:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        rng.normal(size=(8, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES)
    )
    y = np.array([0, 1] * 4)
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))]
    ).fit(x, y)
    artifact = {"pipeline": pipeline, "feature_names": list(FEATURE_NAMES)}
    artifact.update(overrides)
    return artifact


def _valid_preprocessing_artifact(**overrides) -> dict:
    artifact = {
        "scaler": _fitted_scaler(),
        "feature_names": list(FEATURE_NAMES),
        "threshold": 0.5,
    }
    artifact.update(overrides)
    return artifact


def _write_valid_triplet(tmp_path: Path) -> None:
    import joblib

    joblib.dump(_valid_kmeans_artifact(), tmp_path / "kmeans.joblib")
    joblib.dump(_valid_logistic_artifact(), tmp_path / "logistic_regression.joblib")
    joblib.dump(
        _valid_preprocessing_artifact(),
        tmp_path / "neural_network_preprocessing.joblib",
    )


def _write_triplet(
    tmp_path: Path,
    kmeans: dict | None = None,
    logistic: dict | None = None,
    preprocessing: dict | None = None,
) -> None:
    import joblib

    joblib.dump(
        kmeans if kmeans is not None else _valid_kmeans_artifact(),
        tmp_path / "kmeans.joblib",
    )
    joblib.dump(
        logistic if logistic is not None else _valid_logistic_artifact(),
        tmp_path / "logistic_regression.joblib",
    )
    joblib.dump(
        preprocessing if preprocessing is not None else _valid_preprocessing_artifact(),
        tmp_path / "neural_network_preprocessing.joblib",
    )


class _FakeKeras:
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape


# --------------------------------------------------------------------------
# Malformed / incompatible artifact validation
# --------------------------------------------------------------------------

def test_non_dict_artifact_is_rejected(tmp_path):
    import joblib

    _write_triplet(tmp_path, kmeans=[1, 2])

    with pytest.raises(ValueError, match="kmeans.joblib must be a dict"):
        load_models(tmp_path)


def test_missing_required_artifact_key_is_rejected(tmp_path):
    kmeans = _valid_kmeans_artifact()
    kmeans.pop("model")
    _write_triplet(tmp_path, kmeans=kmeans)

    with pytest.raises(ValueError, match="missing required keys"):
        load_models(tmp_path)


def test_wrong_feature_names_are_rejected(tmp_path):
    preprocessing = _valid_preprocessing_artifact(
        feature_names=list(FEATURE_NAMES[:-1]) + ["Wrong"]
    )
    _write_triplet(tmp_path, preprocessing=preprocessing)

    with pytest.raises(ValueError, match="feature_names do not match"):
        load_models(tmp_path)


def test_scaler_with_wrong_feature_count_is_rejected(tmp_path):
    kmeans = _valid_kmeans_artifact(scaler=StandardScaler().fit(np.zeros((4, 5))))
    _write_triplet(tmp_path, kmeans=kmeans)

    with pytest.raises(ValueError, match="does not accept the saved 30 features"):
        load_models(tmp_path)


def test_scaler_with_wrong_feature_order_is_rejected(tmp_path):
    columns = [FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:]]
    kmeans = _valid_kmeans_artifact(scaler=_fitted_scaler(columns=columns))
    _write_triplet(tmp_path, kmeans=kmeans)

    with pytest.raises(ValueError, match="feature order does not match"):
        load_models(tmp_path)


def test_logistic_with_wrong_classes_are_rejected(tmp_path):
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(8, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))]
    ).fit(x, np.array([1, 2, 1, 2, 1, 2, 1, 2]))
    logistic = _valid_logistic_artifact(pipeline=pipeline)
    _write_triplet(tmp_path, logistic=logistic)

    with pytest.raises(ValueError, match=r"classes must be \[0, 1\]"):
        load_models(tmp_path)


def test_kmeans_with_zero_clusters_is_rejected(tmp_path):
    kmeans = _valid_kmeans_artifact()
    kmeans["model"].n_clusters = 0
    _write_triplet(tmp_path, kmeans=kmeans)

    with pytest.raises(ValueError, match="n_clusters must be a positive integer"):
        load_models(tmp_path)


def test_kmeans_with_centers_shaped_mismatch_is_rejected(tmp_path):
    kmeans = _valid_kmeans_artifact()
    kmeans["model"].n_clusters = 5  # inconsistent with cluster_centers_ shape (2, 30)
    _write_triplet(tmp_path, kmeans=kmeans)

    with pytest.raises(ValueError, match=r"cluster_centers_ must have shape \(5, 30\)"):
        load_models(tmp_path)


@pytest.mark.parametrize("bad_threshold", [1.5, -0.1, "0.5", None])
def test_bad_neural_threshold_is_rejected(tmp_path, bad_threshold):
    preprocessing = _valid_preprocessing_artifact(threshold=bad_threshold)
    _write_triplet(tmp_path, preprocessing=preprocessing)

    with pytest.raises(ValueError, match="threshold must be a number between 0 and 1"):
        load_models(tmp_path)


def test_keras_with_wrong_input_shape_is_rejected(tmp_path, monkeypatch):
    _write_valid_triplet(tmp_path)
    monkeypatch.setattr(
        "tensorflow.keras.models.load_model",
        lambda path, compile=False: _FakeKeras((None, 5), (None, 1)),
    )

    with pytest.raises(ValueError, match="input shape"):
        load_models(tmp_path)


def test_keras_with_wrong_output_shape_is_rejected(tmp_path, monkeypatch):
    _write_valid_triplet(tmp_path)
    monkeypatch.setattr(
        "tensorflow.keras.models.load_model",
        lambda path, compile=False: _FakeKeras((None, 30), (None, 2)),
    )

    with pytest.raises(ValueError, match="output_shape"):
        load_models(tmp_path)


# --------------------------------------------------------------------------
# Single-model evaluation and K-Means exclusion from supervised metrics
# --------------------------------------------------------------------------

def test_evaluate_predictions_supports_single_supervised_model(models):
    values = _labelled_values()

    logistic_only = predict_batch(
        models, values, selected_models=("logistic_regression",)
    )
    assert set(logistic_only.columns) == {
        "logistic_regression_class",
        "logistic_regression_probability",
    }
    metrics = evaluate_predictions(values, logistic_only)
    assert set(metrics) == {"logistic_regression"}
    assert set(metrics["logistic_regression"]) == {
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    }

    neural_only = predict_batch(models, values, selected_models=("neural_network",))
    metrics = evaluate_predictions(values, neural_only)
    assert set(metrics) == {"neural_network"}

    both = predict_batch(models, values)
    assert set(evaluate_predictions(values, both)) == {
        "logistic_regression",
        "neural_network",
    }


def test_evaluate_predictions_without_supervised_columns_returns_no_metrics(models):
    values = _labelled_values()
    kmeans_only = predict_batch(models, values, selected_models=("kmeans",))
    assert list(kmeans_only.columns) == ["kmeans_cluster"]

    assert evaluate_predictions(values, kmeans_only) == {}


# --------------------------------------------------------------------------
# K-Means output restrictions
# --------------------------------------------------------------------------

def test_kmeans_predictions_are_cluster_ids_only(models):
    values = _labelled_values().drop(columns=[TARGET_NAME])
    n_clusters = int(models.kmeans["model"].n_clusters)

    kmeans_result = predict_batch(models, values, selected_models=("kmeans",))
    assert list(kmeans_result.columns) == ["kmeans_cluster"]
    assert pd.api.types.is_integer_dtype(kmeans_result["kmeans_cluster"])
    assert kmeans_result["kmeans_cluster"].min() >= 0
    assert kmeans_result["kmeans_cluster"].max() < n_clusters
    assert set(FORBIDDEN_KMEANS_OUTPUT_COLUMNS).isdisjoint(kmeans_result.columns)

    full_result = predict_batch(models, values)
    assert set(FORBIDDEN_KMEANS_OUTPUT_COLUMNS).isdisjoint(full_result.columns)
    # The saved (unvalidated) class mapping is loaded but must not leak into outputs.
    assert "kmeans_class" not in full_result.columns
    assert full_result["kmeans_cluster"].nunique() <= n_clusters

    single = predict(models, {name: float(i) for i, name in enumerate(FEATURE_NAMES)})
    assert "kmeans_cluster" in single
    assert set(FORBIDDEN_KMEANS_OUTPUT_COLUMNS).isdisjoint(single)


# --------------------------------------------------------------------------
# Chunked inference: equivalence, order/index preservation, chunk validation
# --------------------------------------------------------------------------

def test_chunked_predictions_match_full_batch(models, features):
    features.index = [index * 5 + 2 for index in range(len(features))]  # sparse index
    full = predict_batch(models, features)  # single pass (DEFAULT_CHUNK_SIZE >= rows)
    assert DEFAULT_CHUNK_SIZE >= len(features)
    assert list(full.index) == list(features.index)

    for chunk_size in (1, 5, 7):
        chunked = predict_batch(models, features, chunk_size=chunk_size)
        assert list(chunked.columns) == list(full.columns)
        assert list(chunked.index) == list(full.index)
        assert list(chunked.index) == list(features.index)
        for column in full.columns:
            expected = full[column].to_numpy()
            observed = chunked[column].to_numpy()
            if column.endswith("probability"):
                np.testing.assert_allclose(expected, observed, rtol=1e-6, atol=1e-9)
            else:
                np.testing.assert_array_equal(expected, observed)


def test_chunked_predictions_exclude_class_and_reject_bad_chunk_size(models):
    values = _labelled_values()
    chunked = predict_batch(models, values, chunk_size=1)
    assert TARGET_NAME not in chunked.columns
    assert list(chunked.columns) == [
        "kmeans_cluster",
        "logistic_regression_class",
        "logistic_regression_probability",
        "neural_network_class",
        "neural_network_probability",
    ]
    assert list(chunked.index) == list(values.index)

    for bad_chunk_size in (0, -3, 1.5, "8192", True, None):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            predict_batch(models, values, chunk_size=bad_chunk_size)
