# SaaS Churn Analytics

Machine learning pipeline for predicting customer churn in subscription-based SaaS products.

This project uses a synthetic SaaS dataset to build account-level churn features from multiple relational tables, train baseline classifiers, evaluate model performance, and document the workflow in a reproducible notebook report.

## Highlights

- Loads and validates five related SaaS data tables: accounts, subscriptions, feature usage, support tickets, and churn events.
- Builds account-level features while avoiding post-churn data leakage.
- Trains and compares Logistic Regression and Random Forest classifiers.
- Uses a fixed decision threshold selected on the validation split.
- Reports precision, recall, F1, ROC AUC, PR AUC, and confusion matrices.
- Includes unit tests, locked dependencies, raw-data checksums, and a notebook report.

## Results

Current test-set results with `FIXED_THRESHOLD = 0.42`, selected on the validation split:

| Model | Precision | Recall | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.425 | 0.773 | 0.548 | 0.825 | 0.713 |
| Random Forest | 0.789 | 0.682 | 0.732 | 0.804 | 0.715 |

The Random Forest currently gives the strongest threshold-based test performance, while Logistic Regression provides a useful linear baseline with higher churn recall.

## Project Structure

```text
saas-churn-analytics/
  data/
    raw/                 Local Kaggle dataset files, not committed
    models/              Generated model artifacts, not committed
    raw_checksums.txt    Expected SHA-256 hashes for raw CSV files
  docs/
    dataset.md           Dataset description and source notes
  notebooks/
    churn_model_report.ipynb
  scripts/
    check_reproducibility.py
    download_dataset.py
    inspect_data.py
  src/
    data/                CSV loading helpers
    evaluation/          Classification metrics
    features/            Account-level feature engineering
    models/              Training pipeline
  tests/
    test_train_baseline_model.py
```

## Dataset

The project uses the RavenStack synthetic SaaS subscription and churn analytics dataset from Kaggle:

```text
rivalytics/saas-subscription-and-churn-analytics-dataset
```

The dataset is synthetic and contains no real customer PII. More details are documented in `docs/dataset.md`.

Target variable:

```text
accounts.churn_flag
```

Raw data files are treated as local source data and are not committed to Git.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the exact locked environment for reproducible results:

```bash
pip install -r requirements-lock.txt
```

For regular development, the project metadata is also defined in `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

## Data Download

Download the raw Kaggle dataset into `data/raw/`:

```bash
python scripts/download_dataset.py
```

The helper script uses the Kaggle CLI. If the CLI is not configured on your machine, download the public dataset manually from Kaggle and place the CSV files in `data/raw/`.

After downloading, the raw files can be checked against the committed checksums.

## Reproducibility

Verify the current Git checkout, raw CSV checksums, and key package versions:

```bash
python scripts/check_reproducibility.py
```

A passing reproducibility check means that the code, data files, and critical dependency versions match the locked project setup.

## Run Tests

```bash
python -m pytest
```

## Train Models

Run the baseline training pipeline:

```bash
python -m src.models.train_baseline_model
```

This trains both baseline models and writes generated artifacts to `data/models/`.

## Notebook Report

The main portfolio report is available at:

```text
notebooks/churn_model_report.ipynb
```

It walks through the dataset, exploratory analysis, feature engineering, train/validation/test split, model comparison, confusion matrices, and Random Forest feature importance.

## Limitations

- The data is synthetic, so model results should be interpreted as a technical demonstration rather than production evidence.
- The current models are baseline classifiers, not fully tuned production models.
- Model artifacts depend on the locked package versions and should be regenerated after dependency changes.
- A dashboard is not included yet; the notebook is currently the primary presentation layer.

## Next Steps

- Add targeted tests for the snapshot-date feature engineering logic.
- Add a small CI workflow that runs tests and the reproducibility check.
- Add a small Streamlit dashboard after the notebook and pipeline remain stable.
