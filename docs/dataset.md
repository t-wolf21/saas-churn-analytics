# RavenStack Synthetic SaaS Dataset

This project uses the RavenStack synthetic SaaS subscription and churn analytics
dataset from Kaggle:

`rivalytics/saas-subscription-and-churn-analytics-dataset`

Author: River @ Rivalytics

License: MIT-like, fully synthetic, no PII. The dataset may be used or remixed
for educational and portfolio purposes with credit to the original author.

## Tables

The raw dataset contains five CSV files:

| File | Rows | Primary purpose |
| --- | ---: | --- |
| `ravenstack_accounts.csv` | 500 | Account profile and account-level churn flag |
| `ravenstack_subscriptions.csv` | 5,000 | Subscription lifecycle, billing, plan, MRR, ARR |
| `ravenstack_feature_usage.csv` | 25,000 | Product usage events by subscription |
| `ravenstack_support_tickets.csv` | 2,000 | Support workload and customer satisfaction |
| `ravenstack_churn_events.csv` | 600 | Churn dates, reasons, refunds, and feedback |

## Relationships

`accounts.account_id` is the main customer key.

`subscriptions.account_id` links subscriptions to accounts.

`feature_usage.subscription_id` links usage events to subscriptions.

`support_tickets.account_id` links support activity to accounts.

`churn_events.account_id` links churn events to accounts.

## Target

The primary prediction target for this project is:

```text
accounts.churn_flag
```

## Raw Data Policy

Files in `data/raw/` are treated as downloaded source data and are not committed
to Git. Re-run `scripts/download_dataset.py` to recreate them locally.
