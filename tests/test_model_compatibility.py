from pathlib import Path

from model_loader import (
    FEATURE_NAMES,
    TARGET_NAME,
    evaluate_predictions,
    load_models,
    predict,
    predict_batch,
)

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def test_saved_models_load_and_accept_saved_feature_shape():
    models = load_models(MODEL_DIR)
    values = {name: 0.0 for name in FEATURE_NAMES}

    result = predict(models, values)

    assert models.kmeans["model"].n_features_in_ == len(FEATURE_NAMES)
    assert models.logistic_regression["pipeline"].n_features_in_ == len(FEATURE_NAMES)
    assert models.logistic_regression["pipeline"].classes_.tolist() == [0, 1]
    assert models.neural_network.input_shape == (None, len(FEATURE_NAMES))
    assert models.neural_network.output_shape == (None, 1)
    assert set(result) == {
        "kmeans_cluster",
        "logistic_regression_class",
        "logistic_regression_probability",
        "neural_network_class",
        "neural_network_probability",
    }
    assert result["kmeans_cluster"] in {0, 1}
    assert result["logistic_regression_class"] in {0, 1}
    assert 0.0 <= result["logistic_regression_probability"] <= 1.0
    assert models.logistic_regression["pipeline"].predict_proba(
        [list(values.values())]
    ).shape == (1, 2)
    assert result["neural_network_class"] in {0, 1}
    assert 0.0 <= result["neural_network_probability"] <= 1.0


def test_batch_prediction_preserves_row_order_and_has_no_unvalidated_kmeans_class():
    import numpy as np
    import pandas as pd

    models = load_models(MODEL_DIR)
    values = pd.DataFrame(
        np.arange(60, dtype=float).reshape(2, 30),
        columns=FEATURE_NAMES,
        index=[11, 4],
    )

    result = predict_batch(models, values)

    assert list(result.index) == [11, 4]
    assert list(result.columns) == [
        "kmeans_cluster",
        "logistic_regression_class",
        "logistic_regression_probability",
        "neural_network_class",
        "neural_network_probability",
    ]
    assert len(result) == len(values)


def test_batch_prediction_accepts_class_but_excludes_it_from_model_input():
    import pandas as pd

    models = load_models(MODEL_DIR)
    values = pd.DataFrame(
        [{**{name: 0.0 for name in FEATURE_NAMES}, TARGET_NAME: 0},
         {**{name: 1.0 for name in FEATURE_NAMES}, TARGET_NAME: 1}]
    )

    result = predict_batch(models, values)
    metrics = evaluate_predictions(values, result)

    assert len(result) == 2
    assert TARGET_NAME not in result.columns
    assert set(metrics) == {"logistic_regression", "neural_network"}
    assert set(metrics["logistic_regression"]) == {
        "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
        "pr_auc", "confusion_matrix"
    }


def test_batch_prediction_without_class_has_no_metrics():
    import pandas as pd

    models = load_models(MODEL_DIR)
    values = pd.DataFrame([{name: 0.0 for name in FEATURE_NAMES}])

    result = predict_batch(models, values)

    assert evaluate_predictions(values, result) == {}


def test_single_class_metrics_leave_roc_auc_undefined():
    import pandas as pd

    models = load_models(MODEL_DIR)
    values = pd.DataFrame(
        [{**{name: 0.0 for name in FEATURE_NAMES}, TARGET_NAME: 0}]
    )
    result = predict_batch(models, values)

    metrics = evaluate_predictions(values, result)

    assert metrics["logistic_regression"]["roc_auc"] is None
    assert metrics["neural_network"]["roc_auc"] is None


def test_invalid_columns_are_rejected():
    import pandas as pd

    models = load_models(MODEL_DIR)
    invalid = pd.DataFrame(
        [[0.0] * 30], columns=list(FEATURE_NAMES[:-1]) + ["wrong"]
    )

    with pytest.raises(ValueError):
        predict_batch(models, invalid)


def test_non_numeric_input_is_rejected():
    import pandas as pd

    models = load_models(MODEL_DIR)
    invalid = pd.DataFrame(
        [{**{name: 0.0 for name in FEATURE_NAMES}, "Amount": "not numeric"}]
    )

    with pytest.raises(ValueError):
        predict_batch(models, invalid)


def test_invalid_class_labels_are_rejected():
    import pandas as pd

    models = load_models(MODEL_DIR)
    invalid = pd.DataFrame(
        [{**{name: 0.0 for name in FEATURE_NAMES}, TARGET_NAME: 2}]
    )

    with pytest.raises(ValueError, match="Class"):
        predict_batch(models, invalid)


def test_non_finite_features_are_rejected():
    import numpy as np
    import pandas as pd

    models = load_models(MODEL_DIR)
    invalid = pd.DataFrame([{name: 0.0 for name in FEATURE_NAMES}])
    invalid.loc[0, "Amount"] = np.inf

    with pytest.raises(ValueError, match="NaN or infinite"):
        predict_batch(models, invalid)


def test_missing_model_artifact_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path)
