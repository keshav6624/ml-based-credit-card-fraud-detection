from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
FEATURE_NAMES = (
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
)
TARGET_NAME = "Class"
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "neural_network": "Neural Network",
    "kmeans": "K-Means",
}

DEFAULT_CHUNK_SIZE = 8192
KMEANS_CLUSTER_COLUMN = "kmeans_cluster"
# The saved K-Means `cluster_to_class` mapping is unvalidated (it is degenerate in this
# project: {0: 0, 1: 0}), so K-Means may only ever contribute a cluster ID. No derived
# class/label/probability column may be produced for K-Means.
FORBIDDEN_KMEANS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "kmeans_class",
    "kmeans_label",
    "kmeans_probability",
)
# Only these supervised models may appear in `evaluate_predictions` output. K-Means is
# intentionally absent: cluster IDs must never enter supervised metrics.
_SUPERVISED_OUTPUT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("logistic_regression", "logistic_regression_class", "logistic_regression_probability"),
    ("neural_network", "neural_network_class", "neural_network_probability"),
)


@dataclass(frozen=True)
class LoadedModels:
    kmeans: dict[str, Any]
    logistic_regression: dict[str, Any]
    neural_network: Any
    neural_network_preprocessing: dict[str, Any]


def _validate_feature_names(artifact: Any, artifact_name: str) -> None:
    """Feature names must exactly match the frozen, ordered 30-feature schema."""
    names = artifact.get("feature_names") if isinstance(artifact, dict) else None
    if names is None or tuple(names) != FEATURE_NAMES:
        preview = (
            tuple(str(name) for name in tuple(names)[: len(FEATURE_NAMES)])
            if names is not None
            else None
        )
        raise ValueError(
            f"{artifact_name} feature_names do not match the saved schema. Expected the "
            f"{len(FEATURE_NAMES)} ordered features {FEATURE_NAMES[0]}..{FEATURE_NAMES[-1]} "
            f"(Time, V1-V28, Amount); got {preview}"
        )


def _validate_fitted_scaler(scaler: Any, artifact_name: str) -> None:
    """The scaler must be fitted on the exact 30 features, in the exact saved order."""
    count = getattr(scaler, "n_features_in_", None)
    if count is None or int(count) != len(FEATURE_NAMES):
        raise ValueError(
            f"{artifact_name} scaler does not accept the saved {len(FEATURE_NAMES)} "
            f"features (fitted feature count: {count!r})"
        )
    order = getattr(scaler, "feature_names_in_", None)
    if order is None:
        raise ValueError(
            f"{artifact_name} scaler must be fitted on a DataFrame so the saved feature "
            "order can be verified. Restore the artifact from its original pipeline run."
        )
    names = tuple(str(name) for name in order)
    if names != FEATURE_NAMES:
        raise ValueError(
            f"{artifact_name} scaler feature order does not match the saved schema; got "
            f"{names[:4]}... but expected {FEATURE_NAMES[:4]}..."
        )


def _validate_kmeans_artifact(artifact: Any, artifact_name: str) -> None:
    """Validate the K-Means artifact dict. The `cluster_to_class` entry is ignored."""
    if not isinstance(artifact, dict):
        raise ValueError(
            f"{artifact_name} must be a dict with keys scaler/model/feature_names; "
            f"got {type(artifact).__name__}"
        )
    missing = {"scaler", "model", "feature_names"} - set(artifact)
    if missing:
        raise ValueError(
            f"{artifact_name} is missing required keys {sorted(missing)}; restore a valid "
            "artifact from the original pipeline run."
        )
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    _validate_feature_names(artifact, artifact_name)
    if not isinstance(artifact["scaler"], StandardScaler):
        raise ValueError(
            f"{artifact_name} scaler must be a StandardScaler; "
            f"got {type(artifact['scaler']).__name__}"
        )
    _validate_fitted_scaler(artifact["scaler"], artifact_name)
    if not isinstance(artifact["model"], KMeans):
        raise ValueError(
            f"{artifact_name} model must be a fitted sklearn KMeans; got "
            f"{type(artifact['model']).__name__}"
        )
    model = artifact["model"]
    raw_clusters = getattr(model, "n_clusters", None)
    if isinstance(raw_clusters, bool) or not isinstance(raw_clusters, (int, np.integer)):
        raise ValueError(
            f"{artifact_name} K-Means n_clusters must be a positive integer; got {raw_clusters!r}"
        )
    n_clusters = int(raw_clusters)
    if n_clusters < 1:
        raise ValueError(
            f"{artifact_name} K-Means n_clusters must be a positive integer; got {n_clusters}"
        )
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError(
            f"{artifact_name} K-Means model does not accept the saved "
            f"{len(FEATURE_NAMES)} features"
        )
    centers = getattr(model, "cluster_centers_", None)
    centers = np.asarray(centers) if centers is not None else np.empty((0,))
    if centers.shape != (n_clusters, len(FEATURE_NAMES)):
        raise ValueError(
            f"{artifact_name} cluster_centers_ must have shape "
            f"{(n_clusters, len(FEATURE_NAMES))}; got {centers.shape!r}. Restore a valid "
            "artifact from the original pipeline run."
        )
    # NOTE: cluster_to_class (if present) is intentionally not validated and never used:
    # the saved mapping is unvalidated, so K-Means output is cluster IDs only.


def _validate_logistic_artifact(artifact: Any, artifact_name: str) -> None:
    """Validate the Logistic Regression artifact dict and its pipeline."""
    if not isinstance(artifact, dict):
        raise ValueError(
            f"{artifact_name} must be a dict with keys pipeline/feature_names; "
            f"got {type(artifact).__name__}"
        )
    missing = {"pipeline", "feature_names"} - set(artifact)
    if missing:
        raise ValueError(f"{artifact_name} is missing required keys {sorted(missing)}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _validate_feature_names(artifact, artifact_name)
    pipeline = artifact["pipeline"]
    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            f"{artifact_name} pipeline must be a sklearn Pipeline; got {type(pipeline).__name__}"
        )
    named_steps = getattr(pipeline, "named_steps", None)
    scaler_step = named_steps.get("scaler") if hasattr(named_steps, "get") else None
    model_step = named_steps.get("model") if hasattr(named_steps, "get") else None
    if not isinstance(scaler_step, StandardScaler):
        raise ValueError(
            f"{artifact_name} pipeline step 'scaler' must be a fitted StandardScaler; got "
            f"{getattr(scaler_step, '__class__', type(None)).__name__}"
        )
    _validate_fitted_scaler(scaler_step, artifact_name)
    if not isinstance(model_step, LogisticRegression):
        raise ValueError(
            f"{artifact_name} pipeline step 'model' must be a fitted LogisticRegression; got "
            f"{getattr(model_step, '__class__', type(None)).__name__}"
        )
    count = getattr(pipeline, "n_features_in_", None)
    if count is None or int(count) != len(FEATURE_NAMES):
        raise ValueError(
            f"{artifact_name} pipeline does not accept the saved {len(FEATURE_NAMES)} "
            f"features (fitted feature count: {count!r})"
        )
    classes = getattr(model_step, "classes_", None)
    if classes is None:
        raise ValueError(
            f"{artifact_name} logistic regression has no fitted classes_; "
            "restore a valid artifact from the original pipeline run."
        )
    try:
        class_list = [int(label) for label in classes]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{artifact_name} logistic regression classes must be the integers 0 and 1; "
            f"got {tuple(classes)!r}"
        ) from exc
    if class_list != [0, 1]:
        raise ValueError(
            f"{artifact_name} logistic regression classes must be [0, 1]; got {class_list}"
        )


def _validate_neural_preprocessing_artifact(artifact: Any, artifact_name: str) -> None:
    """Validate the neural-network preprocessing dict (scaler + decision threshold)."""
    if not isinstance(artifact, dict):
        raise ValueError(
            f"{artifact_name} must be a dict with keys scaler/feature_names/threshold; "
            f"got {type(artifact).__name__}"
        )
    missing = {"scaler", "feature_names", "threshold"} - set(artifact)
    if missing:
        raise ValueError(f"{artifact_name} is missing required keys {sorted(missing)}")
    from sklearn.preprocessing import StandardScaler

    _validate_feature_names(artifact, artifact_name)
    if not isinstance(artifact["scaler"], StandardScaler):
        raise ValueError(
            f"{artifact_name} scaler must be a StandardScaler; got "
            f"{type(artifact['scaler']).__name__}"
        )
    _validate_fitted_scaler(artifact["scaler"], artifact_name)
    threshold = artifact["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"{artifact_name} threshold must be a number between 0 and 1; got {threshold!r}"
        )
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError(
            f"{artifact_name} threshold must be a number between 0 and 1; got {threshold_value!r}"
        )


def load_models(model_dir: Path = MODEL_DIR) -> LoadedModels:
    """Load and validate the four frozen artifacts in `model_dir` (read-only)."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        hint = ""
        if "sklearn" in sys.modules:
            hint = (
                " Note: on Windows, importing scikit-learn before TensorFlow can break "
                "TensorFlow's DLL initialization. Import tensorflow before sklearn in "
                "the calling process (app.py already does)."
            )
        raise RuntimeError(
            "TensorFlow is required to load models/neural_network.keras. Install the "
            "pinned dependencies from requirements.txt "
            "(example: pip install tensorflow-cpu==2.20.0)." + hint
        ) from exc

    import joblib

    kmeans = joblib.load(model_dir / "kmeans.joblib")
    logistic_regression = joblib.load(model_dir / "logistic_regression.joblib")
    neural_network_preprocessing = joblib.load(
        model_dir / "neural_network_preprocessing.joblib"
    )
    _validate_kmeans_artifact(kmeans, "kmeans.joblib")
    _validate_logistic_artifact(logistic_regression, "logistic_regression.joblib")
    _validate_neural_preprocessing_artifact(
        neural_network_preprocessing, "neural_network_preprocessing.joblib"
    )

    try:
        neural_network = tf.keras.models.load_model(
            model_dir / "neural_network.keras", compile=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Could not load models/neural_network.keras. Verify the file exists and "
            "matches the pinned TensorFlow/Keras versions in requirements.txt. "
            f"Original error: {exc}"
        ) from exc

    input_shape = getattr(neural_network, "input_shape", None)
    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) != 2
        or input_shape[1] != len(FEATURE_NAMES)
    ):
        raise ValueError(
            "neural_network.keras input shape does not match the saved 30-feature schema; "
            f"got input_shape={input_shape!r}"
        )
    output_shape = getattr(neural_network, "output_shape", None)
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or output_shape[1] != 1
    ):
        raise ValueError(
            "neural_network.keras must produce one output probability; "
            f"got output_shape={output_shape!r}"
        )

    return LoadedModels(
        kmeans=kmeans,
        logistic_regression=logistic_regression,
        neural_network=neural_network,
        neural_network_preprocessing=neural_network_preprocessing,
    )


def _validated_input(values: Any) -> tuple[Any, Any]:
    """Return (features in saved order, optional binary target). Class never enters features."""
    import numpy as np
    import pandas as pd

    if not isinstance(values, pd.DataFrame):
        if isinstance(values, dict):
            missing = [name for name in FEATURE_NAMES if name not in values]
            extra = sorted(key for key in values if key not in FEATURE_NAMES)
            if missing or extra:
                raise ValueError(
                    "Prediction input must contain exactly the 30 model features "
                    "(Time, V1-V28, Amount). The optional Class label is accepted only in "
                    "batch DataFrame inputs and must never be passed here. "
                    f"missing={missing}, extra={extra}"
                )
            frame = pd.DataFrame([values], columns=FEATURE_NAMES)
        else:
            try:
                frame = pd.DataFrame([values], columns=FEATURE_NAMES)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Prediction input must provide 30 feature values matching "
                    "Time, V1-V28, Amount"
                ) from exc
        try:
            frame = frame.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("All model features must be numeric") from exc
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Model features cannot contain NaN or infinite values")
        return frame, None

    if values.empty:
        raise ValueError("The uploaded CSV is empty")
    missing = [name for name in FEATURE_NAMES if name not in values.columns]
    allowed = set(FEATURE_NAMES) | {TARGET_NAME}
    extra = [name for name in values.columns if name not in allowed]
    if missing or extra:
        raise ValueError(
            f"Input columns must contain exactly the 30 model features and may "
            f"include optional {TARGET_NAME}; missing={missing}, extra={extra}"
        )

    frame = values.loc[:, list(FEATURE_NAMES)]
    target = None
    if TARGET_NAME in values.columns:
        target = pd.to_numeric(values[TARGET_NAME], errors="coerce")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError(
                f"{TARGET_NAME} must contain only numeric binary labels 0 or 1 "
                "and cannot be missing"
            )
        target = target.astype(np.int64)

    if frame.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features")
    try:
        frame = frame.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("All model features must be numeric") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Model features cannot contain NaN or infinite values")
    return frame, target


def _features(values: dict[str, float] | Any) -> Any:
    return _validated_input(values)[0]


def validate_dataframe(values: Any) -> dict[str, Any]:
    frame, target = _validated_input(values)
    return {
        "rows": len(frame),
        "features": list(frame.columns),
        "has_labels": target is not None,
        "fraud_count": int(target.sum()) if target is not None else None,
    }


def predict(models: LoadedModels, values: dict[str, float] | Any) -> dict[str, Any]:
    """Score a single row. K-Means contributes a cluster ID only."""
    frame = _features(values)
    kmeans_scaled = models.kmeans["scaler"].transform(frame)
    cluster = int(models.kmeans["model"].predict(kmeans_scaled)[0])

    logistic = models.logistic_regression["pipeline"]
    logistic_class = int(logistic.predict(frame)[0])
    logistic_probability = float(logistic.predict_proba(frame)[0, 1])

    preprocessing = models.neural_network_preprocessing
    neural_scaled = preprocessing["scaler"].transform(frame)
    neural_probability = float(models.neural_network.predict(neural_scaled, verbose=0)[0, 0])
    neural_class = int(neural_probability >= float(preprocessing["threshold"]))

    return {
        "kmeans_cluster": cluster,
        "logistic_regression_class": logistic_class,
        "logistic_regression_probability": logistic_probability,
        "neural_network_class": neural_class,
        "neural_network_probability": neural_probability,
    }


def _validated_chunk_size(chunk_size: Any) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer (rows per inference chunk); "
            f"got {chunk_size!r}"
        )
    return int(chunk_size)


def predict_batch(
    models: LoadedModels,
    values: Any,
    selected_models: tuple[str, ...] | list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Any:
    """Score a DataFrame row-by-row in ordered chunks of `chunk_size` rows.

    Row order, the input index, the output column set, and the fixed column order are
    preserved. `chunk_size` only controls how many rows are sent to each model at once;
    it must not change predictions beyond normal floating-point tolerance.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(values, pd.DataFrame):
        raise TypeError("Batch prediction requires a pandas DataFrame")
    chunk_size = _validated_chunk_size(chunk_size)
    frame, _ = _validated_input(values)  # Class is excluded; non-finite values rejected
    selected = set(MODEL_DISPLAY_NAMES) if selected_models is None else set(selected_models)
    unknown = selected - set(MODEL_DISPLAY_NAMES)
    if unknown:
        raise ValueError(f"Unknown model selection: {sorted(unknown)}")
    if not selected:
        raise ValueError("Select at least one model")

    collected: dict[str, list[Any]] = {}

    def _collect(column: str, chunk_values: Any) -> None:
        collected.setdefault(column, []).append(np.asarray(chunk_values))

    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        if "kmeans" in selected:
            kmeans_scaled = models.kmeans["scaler"].transform(chunk)
            _collect(KMEANS_CLUSTER_COLUMN, models.kmeans["model"].predict(kmeans_scaled))
        if "logistic_regression" in selected:
            logistic = models.logistic_regression["pipeline"]
            _collect("logistic_regression_class", logistic.predict(chunk))
            _collect("logistic_regression_probability", logistic.predict_proba(chunk)[:, 1])
        if "neural_network" in selected:
            preprocessing = models.neural_network_preprocessing
            neural_scaled = preprocessing["scaler"].transform(chunk)
            neural_probabilities = models.neural_network.predict(
                neural_scaled, verbose=0
            ).reshape(-1)
            _collect(
                "neural_network_class",
                (neural_probabilities >= float(preprocessing["threshold"])).astype(np.int64),
            )
            _collect("neural_network_probability", neural_probabilities)

    result = pd.DataFrame(index=values.index)
    if "kmeans" in selected:
        result[KMEANS_CLUSTER_COLUMN] = np.concatenate(collected[KMEANS_CLUSTER_COLUMN])
    if "logistic_regression" in selected:
        result["logistic_regression_class"] = np.concatenate(
            collected["logistic_regression_class"]
        )
        result["logistic_regression_probability"] = np.concatenate(
            collected["logistic_regression_probability"]
        )
    if "neural_network" in selected:
        result["neural_network_class"] = np.concatenate(collected["neural_network_class"])
        result["neural_network_probability"] = np.concatenate(
            collected["neural_network_probability"]
        )
    return result


def evaluate_predictions(values: Any, predictions: Any) -> dict[str, dict[str, Any]]:
    """Compute supervised metrics for whichever supervised models were run.

    Logistic Regression and Neural Network metrics are computed independently, so a
    single selected model is enough. K-Means is unsupervised and intentionally excluded:
    its saved class mapping is unvalidated, so cluster IDs must never enter these metrics.
    """
    import pandas as pd
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if not isinstance(values, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
        raise TypeError("Evaluation requires pandas DataFrames")
    _, target = _validated_input(values)
    if target is None:
        return {}
    if len(target) != len(predictions):
        raise ValueError("Predictions and actual labels must have the same row count")

    actual = target.to_numpy()
    metrics: dict[str, dict[str, float | None]] = {}
    for model_name, class_column, probability_column in _SUPERVISED_OUTPUT_SPECS:
        if class_column not in predictions.columns:
            continue  # this model was not run; no metrics can be computed for it
        predicted = predictions[class_column].to_numpy()
        probabilities = (
            predictions[probability_column].to_numpy()
            if probability_column in predictions.columns
            else None
        )
        auc_ready = probabilities is not None and target.nunique() == 2
        metrics[model_name] = {
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(actual, probabilities)) if auc_ready else None
            ),
            "pr_auc": (
                float(average_precision_score(actual, probabilities)) if auc_ready else None
            ),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        }
    return metrics
