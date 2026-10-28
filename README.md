# Fraud Detection Command Center

A Streamlit-based machine learning dashboard for exploring credit-card transactions and running saved fraud-detection models. The application supports CSV validation, batch inference, single-transaction predictions, model comparison, transaction exploration, and artifact health checks.

> **Project status:** Functional MVP. Held-out model performance and full dependency compatibility still require verification. This project is intended for educational and demonstration purposes, not production financial decisions.

## Table of Contents

- [Features](#features)
- [Models](#models)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Input Data Format](#input-data-format)
- [Application Pages](#application-pages)
- [Testing and Verification](#testing-and-verification)
- [Model Evaluation](#model-evaluation)
- [Artifact Integrity](#artifact-integrity)
- [Known Limitations](#known-limitations)
- [Security Considerations](#security-considerations)
- [Future Improvements](#future-improvements)

## Features

- **Batch predictions:** Run selected models against uploaded transaction CSV files.
- **Single-transaction prediction:** Analyze a transaction selected from an uploaded dataset or entered manually.
- **Model comparison:** View supervised classification metrics when ground-truth labels are available.
- **Transaction explorer:** Search and filter processed transactions and download results.
- **Input validation:** Check feature names, required columns, numeric values, labels, and non-finite values.
- **Model health monitoring:** Inspect model loading status, artifact hashes, runtime versions, and model-related warnings.
- **Batch processing:** Process large inputs in chunks while preserving row order.
- **Session management:** Clear previous uploads, predictions, metrics, and manual inputs.
- **CSV templates:** Download templates for unlabeled and labeled transaction data.

## Models

The application loads four saved artifacts representing three model approaches.

| Model | Artifact | Output |
|---|---|---|
| Logistic Regression | `models/logistic_regression.joblib` | Predicted class and model score |
| Neural Network | `models/neural_network.keras` | Predicted class and model score |
| Neural Network preprocessing | `models/neural_network_preprocessing.joblib` | Saved preprocessing and threshold configuration |
| K-Means | `models/kmeans.joblib` | Cluster ID only |

### Logistic Regression

The saved Logistic Regression model is a scikit-learn pipeline containing a `StandardScaler` and a fitted `LogisticRegression` estimator. Its class labels are `[0, 1]`.

The pipeline performs its own scaling. Do not apply an additional scaler before passing input to this pipeline.

### Neural Network

The saved Keras model expects 30 input features and produces a single output. The associated preprocessing artifact stores the preprocessing configuration and decision threshold.

The preprocessing transformation must be applied exactly once. The dashboard uses a threshold of `0.5` as documented in the saved configuration.

The displayed scores should not be interpreted as calibrated real-world fraud risk without separate calibration validation.

### K-Means

K-Means is an unsupervised clustering model. It returns cluster IDs, not verified fraud classifications.

The saved cluster-to-class mapping is unvalidated and degenerate: `{0: 0, 1: 0}`. The application must not use this mapping to classify fraud, produce fraud probabilities, or calculate supervised classification metrics.

## Project Structure

```text
capstone/
├── app.py
├── model_loader.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── kmeans.joblib
│   ├── logistic_regression.joblib
│   ├── neural_network.keras
│   └── neural_network_preprocessing.joblib
├── docs/
│   └── artifact_manifest.json
└── tests/
    ├── test_model_compatibility.py
    ├── test_loader_hardening.py
    └── test_dashboard.py
```

The `app.py` file is the Streamlit entry point. The `model_loader.py` module loads and validates the saved artifacts and handles inference. The `tests/` directory contains model compatibility, input validation, and dashboard tests.

## Requirements

- Python 3.12
- Dependencies specified in `requirements.txt`
- A compatible installation of TensorFlow CPU, Keras, scikit-learn, Streamlit, pandas, NumPy, and joblib
- The original saved model artifacts in the `models/` directory

### Important dependency compatibility warning

The artifacts were saved with scikit-learn `1.6.1`, while the previously tested development environment used scikit-learn `1.4.2`.

The `requirements.txt` file pins scikit-learn `1.6.1`, but the **complete pinned dependency stack has not yet been verified end to end in a fresh environment**. The existing tests passing under the older runtime does not eliminate this compatibility risk.

Follow the installation and verification steps below before relying on the environment.

## Dataset

The original training dataset is not included in Git because of its size.

Download the Credit Card Fraud Detection dataset from Kaggle, extract it, and place creditcard.csv at:

```powershell
data/creditcard.csv
```

## Retraining Models

The repository includes retrain_models.py for model retraining. Retraining requires the training dataset described in the Training Dataset section.

```powershell
python retrain_models.py
```

## Installation

### Windows PowerShell

Open PowerShell in the project directory:

```powershell
cd C:\Users\kesha\Downloads\capstone

py -3.12 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, consult the Python documentation for the appropriate activation method for your system. Avoid changing system-wide execution policies unnecessarily.

### macOS and Linux

```bash
python3.12 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
```

Use the virtual environment for installation, testing, and application execution. Do not upgrade or downgrade packages in another Python environment to resolve project-specific compatibility issues.

## Run the Application

With the virtual environment activated:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit.

To use the dashboard:

1. Navigate to **Fraud Detection**.
2. Upload a transaction CSV with the required schema.
3. Select the models to run.
4. Start prediction.
5. Explore the output, inspect metrics when labels are available, and download the results.

## Input Data Format

The application expects a CSV in the credit-card transaction format, with exactly 30 numeric feature columns and an optional `Class` column.

### Required features

The required feature names are case-sensitive:

```text
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount
```

The feature order is normalized internally to match the saved schema. Column names must still match exactly.

### Optional ground-truth label

The optional `Class` column must contain numeric binary labels:

- `0` — legitimate transaction
- `1` — fraudulent transaction

`Class` is used only as ground truth for evaluation. It must never be passed into a model as an input feature.

If `Class` is absent, predictions can still be generated, but supervised classification metrics cannot be calculated.

### Data validation rules

The input CSV must satisfy the following requirements:

- All 30 required features must be present.
- Feature names must match the expected names.
- Feature values must be numeric and finite.
- Missing required columns and unexpected extra columns are rejected.
- If provided, `Class` must contain only valid binary labels.
- CSV files must be UTF-8 encoded, with or without a byte-order mark (BOM).

`Time` and `Amount` must be provided as raw, unscaled values. The saved model pipelines and preprocessing artifacts handle their expected transformations.

The Fraud Detection page provides downloadable CSV templates for feature-only input and labeled input.

### PCA-transformed features

The `V1`–`V28` columns in the commonly used credit-card fraud dataset are PCA-transformed features, not the original transaction attributes.

For manual prediction, use genuine values from compatible transaction data whenever possible. Arbitrary manually entered values may be outside the distribution on which the models were trained.

## Application Pages

The dashboard provides six pages.

| Page | Purpose |
|---|---|
| **Overview** | Displays headline counts and class distributions after a run. |
| **Fraud Detection** | Uploads and validates data, runs selected models, and provides prediction downloads. |
| **Model Comparison** | Displays per-model supervised metrics and confusion matrices when valid labels are available. |
| **Transaction Explorer** | Searches and filters processed results and supports downloads. |
| **Single Transaction Prediction** | Predicts one transaction using a selected dataset row or manual entry of 30 features. |
| **Model Health** | Reports artifact status, file hashes, runtime versions, feature schema, thresholds, and relevant limitations. |

### Session and result handling

The sidebar includes a Reset / Clear Session control with a confirmation step.

Uploading a different dataset invalidates predictions and metrics associated with the previous upload. The dashboard is designed to avoid silently presenting stale results as though they belong to a new dataset.

## Testing and Verification

Activate the project virtual environment before running the checks.

### 1. Compile the Python modules

```powershell
python -m py_compile app.py model_loader.py
```

### 2. Run the automated tests

```powershell
pytest -q
```

The last recorded test run reported **44 passing tests**. Run the suite again after any changes; this historical result does not guarantee that a new environment will pass.

The existing suites cover:

- Model artifact compatibility
- Invalid or malformed artifacts
- Input schema and value validation
- Batch inference and chunking behavior
- K-Means output restrictions
- Dashboard empty states and single-transaction prediction
- CSV parsing and validation
- Session reset and stale-result invalidation
- Artifact manifest checks

### 3. Check dependency consistency

```powershell
pip check
```

A clean result indicates that pip did not identify broken declared dependencies. It does not prove that serialized artifacts or the complete runtime stack are compatible.

### 4. Verify artifact loading

```powershell
python -c "import tensorflow; from model_loader import load_models; load_models(); print('Artifacts loaded successfully')"
```

Importing TensorFlow first is intentional because the previously tested Windows environment exhibited a TensorFlow DLL initialization issue when scikit-learn was imported first.

Run this check only after installing the intended dependency versions. Do not treat successful loading alone as proof of model quality.

### 5. Start the dashboard

```powershell
streamlit run app.py
```

Manually visit all six pages, upload a valid CSV, run predictions, inspect the results, and test the download and reset controls. Also test at least one invalid upload.

Automated tests and a successful health endpoint do not replace end-to-end manual verification.

## Model Evaluation

The Model Comparison page calculates supervised metrics from the uploaded labeled dataset. In the current implementation, these metrics are **in-sample**: they describe predictions on the supplied rows and must not be interpreted as evidence of generalization to unseen transactions.

Before claiming reliable fraud-detection performance, evaluate Logistic Regression and the Neural Network on a genuinely held-out dataset that was not used for training or threshold tuning.

Recommended metrics include:

- Precision
- Recall
- F1-score
- Balanced accuracy
- Confusion matrix
- Average precision / PR-AUC
- ROC-AUC, where appropriate

Fraud detection often involves severe class imbalance, so accuracy alone is insufficient.

Document the dataset source, split methodology, class distribution, thresholds, and actual measured results. If the original training data or training history is unavailable, explicitly disclose that the independence of the test set cannot be established.

## Artifact Integrity

The four saved model artifacts are treated as frozen files. Do not retrain, modify, rename, move, overwrite, or regenerate them as a way to resolve dependency warnings.

Their SHA-256 hashes are recorded in:

```text
docs/artifact_manifest.json
```

### Verify hashes on Windows

```powershell
Get-ChildItem models -File | Get-FileHash -Algorithm SHA256
```

Compare the output against the manifest.

To check the manifest programmatically:

```powershell
python -c "import hashlib,json; m=json.load(open('docs/artifact_manifest.json',encoding='utf-8')); [print(a['path'], 'MATCH' if hashlib.sha256(open(a['path'],'rb').read()).hexdigest()==a['sha256'] else 'MISMATCH') for a in m['artifacts']]"
```

On macOS or Linux:

```bash
sha256sum models/*
```

A hash mismatch should be investigated before using the affected artifact.

## Known Limitations

- **Dependency compatibility:** The complete pinned stack must still be verified in a fresh environment against the saved artifacts.
- **No confirmed held-out performance:** Existing in-sample metrics do not establish performance on unseen transactions.
- **Uncalibrated model scores:** Displayed scores are not guaranteed to represent calibrated fraud probabilities.
- **Unvalidated K-Means mapping:** K-Means provides cluster IDs only and is excluded from supervised fraud classification.
- **Frozen artifacts:** Model retraining and threshold optimization are not part of the current workflow.
- **Dataset dependence:** Predictions are meaningful only when inputs follow the expected schema and are compatible with the model's training data.
- **Educational scope:** The application does not establish production-grade fraud prevention, financial decision-making, or regulatory compliance.

## Security Considerations

- Only load model artifacts from trusted sources. Loading untrusted joblib/pickle-based files can execute malicious code.
- Do not upload confidential transaction data to an environment unless its data-handling and access controls are appropriate.
- Avoid committing private datasets, credentials, secrets, or personally identifiable information to version control.
- Validate uploaded files before inference and consider upload-size and resource limits before exposing the application publicly.
- A local Streamlit dashboard is not, by itself, a secured production service.

## Future Improvements

Potential next steps include:

- Resolve and document the full dependency compatibility matrix.
- Establish a reproducible, independent held-out evaluation.
- Add documented threshold analysis using validation data.
- Add data-drift and model-performance monitoring if a reliable data source becomes available.
- Add deployment-specific access controls and resource limits.
- Record experiment metadata and model provenance without modifying the frozen artifacts.

---

## Project Summary

The Fraud Detection Command Center provides a dashboard for running saved Logistic Regression, Neural Network, and K-Means artifacts against compatible transaction CSV files.

Its current focus is reliable input validation, controlled inference, transparent artifact handling, and clear presentation of model outputs. Dependency compatibility and independent model evaluation remain necessary steps before stronger claims about reliability or production readiness can be made.