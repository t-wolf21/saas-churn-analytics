# saas-churn-analytics

Machine learning pipeline and analytics dashboard for predicting customer churn in subscription-based SaaS products.

## Quick Start

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Load a CSV file from the local raw data folder:

```python
from src.data.load_data import load_csv

df = load_csv("churn.csv")
```

Load a committed sample dataset for demos and testing:

```python
from src.data.load_data import load_csv

df = load_csv("churn_sample.csv", sample=True)
```

## Data Folders

```text
data/
  raw/        Local raw datasets. Do not commit real customer data.
  sample/     Small demo datasets that make the project easy to test.
  processed/  Generated cleaned datasets.
  models/     Trained model artifacts.
```

Expected CSV target column:

```text
churn
```
