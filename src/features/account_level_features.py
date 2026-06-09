from __future__ import annotations

import pandas as pd


def _parse_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = df.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    return parsed


def _parse_boolean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = df.copy()
    for column in columns:
        if column in parsed.columns:
            normalized = parsed[column].astype(str).str.strip().str.lower()
            parsed[column] = normalized.map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False,
                }
            ).fillna(False)
    return parsed


def _infer_observation_end_date(
    accounts: pd.DataFrame,
    subscriptions: pd.DataFrame,
    support_tickets: pd.DataFrame,
    feature_usage: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> pd.Timestamp:
    date_series: list[pd.Series] = []

    for frame, columns in (
        (accounts, ["signup_date"]),
        (subscriptions, ["start_date", "end_date"]),
        (support_tickets, ["submitted_at", "closed_at"]),
        (feature_usage, ["usage_date"]),
        (churn_events, ["churn_date"]),
    ):
        for column in columns:
            if column in frame.columns:
                date_series.append(pd.to_datetime(frame[column], errors="coerce"))

    combined_dates = pd.concat(date_series, ignore_index=True).dropna()
    if combined_dates.empty:
        raise ValueError("Unable to infer an observation end date from the provided tables.")

    return combined_dates.max().normalize()


def _build_snapshot_dates(
    accounts: pd.DataFrame,
    subscriptions: pd.DataFrame,
    churn_events: pd.DataFrame,
    observation_end_date: pd.Timestamp,
) -> pd.DataFrame:
    snapshot_dates = accounts[["account_id", "signup_date", "churn_flag"]].copy()
    snapshot_dates["signup_date"] = pd.to_datetime(snapshot_dates["signup_date"], errors="coerce")
    snapshot_dates["churn_flag"] = snapshot_dates["churn_flag"].fillna(False).astype(bool)

    if churn_events.empty:
        event_churn_dates = pd.DataFrame(columns=["account_id", "event_churn_date"])
    else:
        event_churn_dates = (
            churn_events.groupby("account_id", as_index=False)["churn_date"]
            .min()
            .rename(columns={"churn_date": "event_churn_date"})
        )

    subscription_churn_dates = subscriptions.loc[
        subscriptions["churn_flag"] & subscriptions["end_date"].notna(),
        ["account_id", "end_date"],
    ]
    if subscription_churn_dates.empty:
        subscription_churn_dates = pd.DataFrame(columns=["account_id", "subscription_churn_date"])
    else:
        subscription_churn_dates = (
            subscription_churn_dates.groupby("account_id", as_index=False)["end_date"]
            .min()
            .rename(columns={"end_date": "subscription_churn_date"})
        )

    snapshot_dates = snapshot_dates.merge(event_churn_dates, on="account_id", how="left")
    snapshot_dates = snapshot_dates.merge(subscription_churn_dates, on="account_id", how="left")
    snapshot_dates["churn_date"] = snapshot_dates[["event_churn_date", "subscription_churn_date"]].min(axis=1)

    snapshot_dates["snapshot_date"] = observation_end_date
    churned_accounts = snapshot_dates["churn_flag"] & snapshot_dates["churn_date"].notna()
    snapshot_dates.loc[churned_accounts, "snapshot_date"] = snapshot_dates.loc[churned_accounts, "churn_date"] - pd.Timedelta(
        days=1
    )

    snapshot_dates["account_age_days"] = (snapshot_dates["snapshot_date"] - snapshot_dates["signup_date"]).dt.days
    snapshot_dates["account_age_days"] = snapshot_dates["account_age_days"].clip(lower=0)

    return snapshot_dates[["account_id", "snapshot_date", "account_age_days"]]


def _filter_subscriptions_to_snapshot(subscriptions: pd.DataFrame, snapshot_dates: pd.DataFrame) -> pd.DataFrame:
    filtered = subscriptions.merge(
        snapshot_dates[["account_id", "snapshot_date"]],
        on="account_id",
        how="left",
    )
    cutoff = filtered["snapshot_date"] + pd.Timedelta(days=1)
    keep_rows = (
        filtered["start_date"].notna()
        & filtered["snapshot_date"].notna()
        & (filtered["start_date"] < cutoff)
    )
    return filtered.loc[keep_rows].copy()


def build_subscription_features(subscriptions: pd.DataFrame) -> pd.DataFrame:
    subscription_features = subscriptions.groupby("account_id").agg(
        avg_mrr=("mrr_amount", "mean"),
        max_mrr=("mrr_amount", "max"),
        avg_arr=("arr_amount", "mean"),
        max_arr=("arr_amount", "max"),
        total_arr=("arr_amount", "sum"),
        avg_seats=("seats", "mean"),
        max_seats=("seats", "max"),
        subscription_count=("subscription_id", "count"),
        has_upgrade=("upgrade_flag", "any"),
        has_downgrade=("downgrade_flag", "any"),
        auto_renew_rate=("auto_renew_flag", "mean"),
        trial_rate=("is_trial", "mean"),
        monthly_billing_rate=("billing_frequency", lambda x: (x == "monthly").mean()),
        annual_billing_rate=("billing_frequency", lambda x: (x == "annual").mean()),
    ).reset_index()
    return subscription_features


def build_support_features(support_tickets: pd.DataFrame, snapshot_dates: pd.DataFrame) -> pd.DataFrame:
    support_tickets = support_tickets.merge(
        snapshot_dates[["account_id", "snapshot_date"]],
        on="account_id",
        how="left",
    )

    cutoff = support_tickets["snapshot_date"] + pd.Timedelta(days=1)

    submitted_tickets = support_tickets.loc[
        support_tickets["submitted_at"].notna()
        & support_tickets["snapshot_date"].notna()
        & (support_tickets["submitted_at"] < cutoff)
    ].copy()

    closed_tickets = support_tickets.loc[
        support_tickets["closed_at"].notna()
        & support_tickets["snapshot_date"].notna()
        & (support_tickets["closed_at"] < cutoff)
    ].copy()

    submission_features = submitted_tickets.groupby("account_id").agg(
        ticket_count=("ticket_id", "count"),
        urgent_ticket_rate=("priority", lambda x: (x == "urgent").mean()),
        high_priority_ticket_rate=("priority", lambda x: (x == "high").mean()),
        medium_priority_ticket_rate=("priority", lambda x: (x == "medium").mean()),
        low_priority_ticket_rate=("priority", lambda x: (x == "low").mean()),
    ).reset_index()

    resolution_features = closed_tickets.groupby("account_id").agg(
        avg_resolution_hours=("resolution_time_hours", "mean"),
        max_resolution_hours=("resolution_time_hours", "max"),
        avg_first_response_minutes=("first_response_time_minutes", "mean"),
        avg_satisfaction_score=("satisfaction_score", "mean"),
        min_satisfaction_score=("satisfaction_score", "min"),
        escalation_rate=("escalation_flag", "mean"),
        has_escalation=("escalation_flag", "any"),
    ).reset_index()

    support_features = submission_features.merge(resolution_features, on="account_id", how="outer")
    return support_features


def build_feature_usage_features(
    feature_usage: pd.DataFrame, subscriptions: pd.DataFrame, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    usage_with_accounts = feature_usage.merge(
        subscriptions[["subscription_id", "account_id"]],
        on="subscription_id",
        how="inner",
    )
    usage_with_accounts = usage_with_accounts.merge(
        snapshot_dates[["account_id", "snapshot_date"]],
        on="account_id",
        how="left",
    )

    cutoff = usage_with_accounts["snapshot_date"] + pd.Timedelta(days=1)
    usage_with_accounts = usage_with_accounts.loc[
        usage_with_accounts["usage_date"].notna()
        & usage_with_accounts["snapshot_date"].notna()
        & (usage_with_accounts["usage_date"] < cutoff)
    ].copy()

    feature_usage_features = usage_with_accounts.groupby("account_id").agg(
        total_usage_count=("usage_count", "sum"),
        avg_usage_count=("usage_count", "mean"),
        total_usage_duration_secs=("usage_duration_secs", "sum"),
        avg_usage_duration_secs=("usage_duration_secs", "mean"),
        unique_features_used=("feature_name", "nunique"),
        total_error_count=("error_count", "sum"),
        avg_error_count=("error_count", "mean"),
        beta_feature_rate=("is_beta_feature", "mean"),
        usage_event_count=("usage_id", "count"),
    ).reset_index()

    feature_usage_features["has_usage"] = feature_usage_features["usage_event_count"] > 0

    return feature_usage_features


def build_account_level_features(
    accounts: pd.DataFrame,
    subscriptions: pd.DataFrame,
    support_tickets: pd.DataFrame,
    feature_usage: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> pd.DataFrame:
    accounts = _parse_datetime_columns(accounts, ["signup_date"])
    accounts = _parse_boolean_columns(accounts, ["is_trial", "churn_flag"])

    subscriptions = _parse_datetime_columns(subscriptions, ["start_date", "end_date"])
    subscriptions = _parse_boolean_columns(
        subscriptions,
        ["is_trial", "upgrade_flag", "downgrade_flag", "churn_flag", "auto_renew_flag"],
    )

    support_tickets = _parse_datetime_columns(support_tickets, ["submitted_at", "closed_at"])
    support_tickets = _parse_boolean_columns(support_tickets, ["escalation_flag"])

    feature_usage = _parse_datetime_columns(feature_usage, ["usage_date"])
    feature_usage = _parse_boolean_columns(feature_usage, ["is_beta_feature"])

    churn_events = _parse_datetime_columns(churn_events, ["churn_date"])

    observation_end_date = _infer_observation_end_date(
        accounts=accounts,
        subscriptions=subscriptions,
        support_tickets=support_tickets,
        feature_usage=feature_usage,
        churn_events=churn_events,
    )

    snapshot_dates = _build_snapshot_dates(
        accounts=accounts,
        subscriptions=subscriptions,
        churn_events=churn_events,
        observation_end_date=observation_end_date,
    )

    subscriptions_at_snapshot = _filter_subscriptions_to_snapshot(subscriptions, snapshot_dates)

    subscription_features = build_subscription_features(subscriptions_at_snapshot)
    support_features = build_support_features(support_tickets, snapshot_dates)
    usage_features = build_feature_usage_features(feature_usage, subscriptions_at_snapshot, snapshot_dates)

    account_level_features = (
        accounts.merge(snapshot_dates, on="account_id", how="left")
        .merge(subscription_features, on="account_id", how="left")
        .merge(usage_features, on="account_id", how="left")
        .merge(support_features, on="account_id", how="left")
    )

    account_level_features = account_level_features.drop(columns=["signup_date", "snapshot_date"], errors="ignore")

    count_columns = [
        "ticket_count",
        "usage_event_count",
        "subscription_count",
        "total_usage_count",
        "total_usage_duration_secs",
        "total_error_count",
        "unique_features_used",
    ]

    rate_columns = [
        "escalation_rate",
        "urgent_ticket_rate",
        "high_priority_ticket_rate",
        "medium_priority_ticket_rate",
        "low_priority_ticket_rate",
        "auto_renew_rate",
        "trial_rate",
        "monthly_billing_rate",
        "annual_billing_rate",
        "beta_feature_rate",
    ]

    flag_columns = [
        "has_escalation",
        "has_upgrade",
        "has_downgrade",
        "has_usage",
    ]

    account_level_features[count_columns] = account_level_features[count_columns].fillna(0)
    account_level_features[rate_columns] = account_level_features[rate_columns].fillna(0)
    for column in flag_columns:
        account_level_features[column] = account_level_features[column].astype("boolean").fillna(False).astype(bool)

    account_level_features["tickets_per_month"] = (
            account_level_features["ticket_count"]
            / ((account_level_features["account_age_days"] / 30).clip(lower=1))
    )

    account_level_features["usage_per_day"] = (
            account_level_features["total_usage_count"]
            / account_level_features["account_age_days"].clip(lower=1)
    )

    account_level_features["error_rate"] = (
            account_level_features["total_error_count"]
            / account_level_features["total_usage_count"].clip(lower=1)
    )

    account_level_features["account_activity_score"] = (
            account_level_features["usage_per_day"]
            * account_level_features["unique_features_used"]
    )

    account_level_features["errors_per_day"] = (
            account_level_features["total_error_count"]
            / account_level_features["account_age_days"].clip(lower=1)
    )

    account_level_features["tickets_per_usage"] = (
            account_level_features["ticket_count"]
            / account_level_features["total_usage_count"].clip(lower=1)
    )

    account_level_features["tickets_per_feature"] = (
            account_level_features["ticket_count"]
            / account_level_features["unique_features_used"].clip(lower=1)
    )

    return account_level_features
