
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
STAGING_DIR = BASE_DIR / "staging_models"

TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    original_rows = len(df)
    original_duplicates = int(df.duplicated().sum())

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicate rows removed: {original_duplicates:,}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Rows after deduplication: {len(df):,}")
    print("\nClass counts after deduplication:")
    print(df[TARGET].value_counts().sort_index())

    missing = df[FEATURES + [TARGET]].isnull().sum().sum()
    if missing:
        raise ValueError(f"Dataset contains {missing} missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    # Audit feature-identical rows that remain after full-row deduplication.
    feature_duplicates = int(df.duplicated(subset=FEATURES, keep=False).sum())
    print(
        "\nRows belonging to repeated feature vectors "
        f"(possible repeated transactions): {feature_duplicates:,}"
    )

    X = df[FEATURES].astype("float32")
    y = df[TARGET].astype("int32")

    # Keep the test set separate from training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print("Test class counts:")
    print(y_test.value_counts().sort_index())

    report = {
        "dataset": str(DATA_PATH),
        "original_rows": int(original_rows),
        "exact_duplicates_removed": int(original_duplicates),
        "rows_after_deduplication": int(len(df)),
        "repeated_feature_vector_rows": int(feature_duplicates),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "models": {},
        "caveat": (
            "Stratified random split. Repeated feature vectors may still "
            "occur across train and test sets; metrics should be interpreted "
            "with this limitation in mind."
        ),
    }

    print("\nTraining Logistic Regression...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    lr_pipeline.fit(X_train, y_train)
    lr_scores = lr_pipeline.predict_proba(X_test)[:, 1]

    # Use the default threshold initially; threshold tuning must use validation data.
    lr_predictions = (lr_scores >= 0.5).astype(int)

    lr_artifact = {
        "pipeline": lr_pipeline,
        "feature_names": FEATURES,
    }
    joblib.dump(
        lr_artifact,
        STAGING_DIR / "logistic_regression.joblib",
    )

    report["models"]["logistic_regression"] = evaluate_model(
        y_test, lr_predictions, lr_scores
    )

    print("\nPreparing Neural Network...")
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    nn_scaler = StandardScaler()
    X_train_scaled = nn_scaler.fit_transform(X_train).astype("float32")
    X_test_scaled = nn_scaler.transform(X_test).astype("float32")

    nn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    nn.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    class_counts = y_train.value_counts()
    negatives = int(class_counts.get(0, 0))
    positives = int(class_counts.get(1, 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both target classes.")

    class_weights = {
        0: 1.0,
        1: negatives / positives,
    }

    nn.fit(
        X_train_scaled,
        y_train.to_numpy(),
        validation_split=0.15,
        epochs=30,
        batch_size=2048,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1,
    )

    nn_scores = nn.predict(X_test_scaled, verbose=0).ravel()
    nn_predictions = (nn_scores >= 0.5).astype(int)

    nn.save(STAGING_DIR / "neural_network.keras")

    joblib.dump(
        {
            "scaler": nn_scaler,
            "feature_names": FEATURES,
            "threshold": 0.5,
        },
        STAGING_DIR / "neural_network_preprocessing.joblib",
    )

    report["models"]["neural_network"] = evaluate_model(
        y_test, nn_predictions, nn_scores
    )

    report_path = STAGING_DIR / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nHeld-out test results:")
    for name, metrics in report["models"].items():
        print(f"\n{name}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Average precision: {metrics['average_precision']:.4f}")
        print("Confusion matrix:")
        print(np.array(metrics["confusion_matrix"]))

    print("\nTraining complete.")
    print(f"Staged artifacts: {STAGING_DIR}")
    print(f"Report: {report_path}")
    print("The existing models folder was not modified.")


def evaluate_model(y_true, predictions, scores):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)

        return {
            "precision": float(
                precision_score(y_true, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, predictions, zero_division=0)
            ),
            "average_precision": float(
                average_precision_score(y_true, scores)
            ),
            "confusion_matrix": confusion_matrix(
                y_true, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_true,
                predictions,
                labels=[0, 1],
                target_names=["legitimate", "fraud"],
                output_dict=True,
                zero_division=0,
            ),
        }


if __name__ == "__main__":
    main()
