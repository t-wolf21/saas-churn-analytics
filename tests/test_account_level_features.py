import numpy as np
import pandas as pd

from src.features import account_level_features as alf


def test_parse_boolean_columns_normalizes_common_values():
    df = pd.DataFrame(
        {
            "flag": ["True", " false ", "1", "0", "yes", "no", None, "unknown"],
        }
    )

    parsed = alf._parse_boolean_columns(df, ["flag"])

    assert parsed["flag"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]


def test_build_snapshot_dates_uses_day_before_earliest_churn_date():
    accounts = pd.DataFrame(
        {
            "account_id": ["A", "B", "C"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01"]),
            "churn_flag": [True, True, False],
        }
    )
    subscriptions = pd.DataFrame(
        {
            "account_id": ["A", "B", "C"],
            "end_date": pd.to_datetime(["2024-03-10", "2024-04-01", None]),
            "churn_flag": [True, True, False],
        }
    )
    churn_events = pd.DataFrame(
        {
            "account_id": ["A"],
            "churn_date": pd.to_datetime(["2024-03-05"]),
        }
    )

    snapshot_dates = alf._build_snapshot_dates(
        accounts=accounts,
        subscriptions=subscriptions,
        churn_events=churn_events,
        observation_end_date=pd.Timestamp("2024-04-30"),
    ).set_index("account_id")

    assert snapshot_dates.loc["A", "snapshot_date"] == pd.Timestamp("2024-03-04")
    assert snapshot_dates.loc["B", "snapshot_date"] == pd.Timestamp("2024-03-31")
    assert snapshot_dates.loc["C", "snapshot_date"] == pd.Timestamp("2024-04-30")
    assert snapshot_dates.loc["A", "account_age_days"] == 63


def test_filter_subscriptions_to_snapshot_excludes_future_subscriptions():
    subscriptions = pd.DataFrame(
        {
            "subscription_id": ["S-before", "S-same-day", "S-after"],
            "account_id": ["A", "A", "A"],
            "start_date": pd.to_datetime(["2024-02-01", "2024-02-10", "2024-02-11"]),
        }
    )
    snapshot_dates = pd.DataFrame(
        {
            "account_id": ["A"],
            "snapshot_date": pd.to_datetime(["2024-02-10"]),
        }
    )

    filtered = alf._filter_subscriptions_to_snapshot(subscriptions, snapshot_dates)

    assert filtered["subscription_id"].tolist() == ["S-before", "S-same-day"]


def test_build_support_features_excludes_post_snapshot_activity():
    snapshot_dates = pd.DataFrame(
        {
            "account_id": ["A"],
            "snapshot_date": pd.to_datetime(["2024-02-10"]),
        }
    )
    support_tickets = pd.DataFrame(
        {
            "ticket_id": ["T-before", "T-open", "T-after"],
            "account_id": ["A", "A", "A"],
            "submitted_at": pd.to_datetime(["2024-02-09", "2024-02-08", "2024-02-11"]),
            "closed_at": pd.to_datetime(["2024-02-09 03:00:00", "2024-02-12 00:00:00", "2024-02-11 00:00:00"]),
            "resolution_time_hours": [3.0, 96.0, 1.0],
            "priority": ["urgent", "low", "high"],
            "first_response_time_minutes": [20.0, 40.0, 5.0],
            "satisfaction_score": [4.0, 1.0, 5.0],
            "escalation_flag": [True, False, True],
        }
    )

    features = alf.build_support_features(support_tickets, snapshot_dates).set_index("account_id")

    assert features.loc["A", "ticket_count"] == 2
    assert features.loc["A", "urgent_ticket_rate"] == 0.5
    assert features.loc["A", "low_priority_ticket_rate"] == 0.5
    assert features.loc["A", "avg_resolution_hours"] == 3.0
    assert features.loc["A", "avg_satisfaction_score"] == 4.0
    assert features.loc["A", "escalation_rate"] == 1.0


def test_build_feature_usage_features_excludes_post_snapshot_usage():
    snapshot_dates = pd.DataFrame(
        {
            "account_id": ["A"],
            "snapshot_date": pd.to_datetime(["2024-02-10"]),
        }
    )
    subscriptions = pd.DataFrame(
        {
            "subscription_id": ["S-1"],
            "account_id": ["A"],
        }
    )
    feature_usage = pd.DataFrame(
        {
            "usage_id": ["U-before", "U-after", "U-other-subscription"],
            "subscription_id": ["S-1", "S-1", "S-unknown"],
            "usage_date": pd.to_datetime(["2024-02-09", "2024-02-11", "2024-02-09"]),
            "feature_name": ["reporting", "exports", "automation"],
            "usage_count": [10, 99, 50],
            "usage_duration_secs": [120, 999, 500],
            "error_count": [2, 9, 5],
            "is_beta_feature": [True, False, True],
        }
    )

    features = alf.build_feature_usage_features(
        feature_usage=feature_usage,
        subscriptions=subscriptions,
        snapshot_dates=snapshot_dates,
    ).set_index("account_id")

    assert features.loc["A", "usage_event_count"] == 1
    assert features.loc["A", "total_usage_count"] == 10
    assert features.loc["A", "total_error_count"] == 2
    assert features.loc["A", "unique_features_used"] == 1
    assert features.loc["A", "beta_feature_rate"] == 1.0
    assert features.loc["A", "has_usage"]


def test_build_account_level_features_fills_missing_values_and_derives_rates():
    accounts = pd.DataFrame(
        {
            "account_id": ["A", "B"],
            "account_name": ["Alpha", "Beta"],
            "industry": ["FinTech", "EdTech"],
            "country": ["US", "DE"],
            "signup_date": ["2024-01-01", "2024-01-10"],
            "referral_source": ["paid", "organic"],
            "plan_tier": ["Pro", "Basic"],
            "seats": [10, 2],
            "is_trial": [False, True],
            "churn_flag": [False, False],
        }
    )
    subscriptions = pd.DataFrame(
        {
            "subscription_id": ["S-1"],
            "account_id": ["A"],
            "start_date": ["2024-01-01"],
            "end_date": [None],
            "plan_tier": ["Pro"],
            "seats": [10],
            "mrr_amount": [100.0],
            "arr_amount": [1200.0],
            "is_trial": [False],
            "upgrade_flag": [True],
            "downgrade_flag": [False],
            "churn_flag": [False],
            "billing_frequency": ["annual"],
            "auto_renew_flag": [True],
        }
    )
    support_tickets = pd.DataFrame(
        {
            "ticket_id": ["T-1"],
            "account_id": ["A"],
            "submitted_at": ["2024-01-15"],
            "closed_at": ["2024-01-16"],
            "resolution_time_hours": [24.0],
            "priority": ["high"],
            "first_response_time_minutes": [30.0],
            "satisfaction_score": [3.0],
            "escalation_flag": [False],
        }
    )
    feature_usage = pd.DataFrame(
        {
            "usage_id": ["U-1"],
            "subscription_id": ["S-1"],
            "usage_date": ["2024-01-20"],
            "feature_name": ["reporting"],
            "usage_count": [20],
            "usage_duration_secs": [300],
            "error_count": [2],
            "is_beta_feature": [False],
        }
    )
    churn_events = pd.DataFrame(columns=["account_id", "churn_date"])

    features = alf.build_account_level_features(
        accounts=accounts,
        subscriptions=subscriptions,
        support_tickets=support_tickets,
        feature_usage=feature_usage,
        churn_events=churn_events,
    ).set_index("account_id")

    assert features.loc["B", "subscription_count"] == 0
    assert features.loc["B", "ticket_count"] == 0
    assert features.loc["B", "total_usage_count"] == 0
    assert not features.loc["B", "has_usage"]
    assert not features.loc["B", "has_escalation"]

    assert features.loc["A", "total_usage_count"] == 20
    assert features.loc["A", "total_error_count"] == 2
    assert features.loc["A", "ticket_count"] == 1
    assert np.isclose(features.loc["A", "error_rate"], 0.1)
    assert np.isclose(features.loc["A", "tickets_per_usage"], 0.05)
    assert np.isclose(features.loc["A", "usage_per_day"], 20 / 19)
