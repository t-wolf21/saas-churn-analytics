import pandas as pd

def build_subscription_features(subscriptions: pd.DataFrame) -> pd.DataFrame:
    subscriptions = subscriptions.copy()
    subscriptions = subscriptions.groupby("account_id").agg(
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
    return subscriptions

def build_support_features(support_tickets: pd.DataFrame) -> pd.DataFrame:
    support_tickets = support_tickets.copy()
    support_tickets = support_tickets.groupby("account_id").agg(
        ticket_count=("ticket_id", "count"),
        avg_resolution_hours=("resolution_time_hours", "mean"),
        max_resolution_hours=("resolution_time_hours", "max"),
        avg_first_response_minutes=("first_response_time_minutes", "mean"),
        avg_satisfaction_score=("satisfaction_score", "mean"),
        min_satisfaction_score=("satisfaction_score", "min"),
        escalation_rate=("escalation_flag", "mean"),
        has_escalation=("escalation_flag", "any"),
        urgent_ticket_rate=("priority", lambda x: (x == "urgent").mean()),
        high_priority_ticket_rate=("priority", lambda x: (x == "high").mean()),
        medium_priority_ticket_rate=("priority", lambda x: (x == "medium").mean()),
        low_priority_ticket_rate=("priority", lambda x: (x == "low").mean()),
    ).reset_index()
    return support_tickets

def build_feature_usage_features(feature_usage: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    feature_usage = feature_usage.copy()
    usage_with_accounts = feature_usage.merge(
        subscriptions[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left",
    )
    feature_usage = usage_with_accounts.groupby("account_id").agg(
        total_usage_count=("usage_count", "sum"),
        avg_usage_count=("usage_count", "mean"),
        total_usage_duration_secs=("usage_duration_secs", "sum"),
        avg_usage_duration_secs=("usage_duration_secs", "mean"),
        unique_features_used=("feature_name", "nunique"),
        total_error_count=("error_count", "sum"),
        avg_error_count=("error_count", "mean"),
        beta_feature_rate=("is_beta_feature", "mean"),
        usage_event_count=("usage_id", "count")
    ).reset_index()

    feature_usage["has_usage"] = feature_usage["usage_event_count"] > 0

    return feature_usage

def build_account_level_features(
    accounts: pd.DataFrame, subscriptions: pd.DataFrame, support_tickets: pd.DataFrame, feature_usage: pd.DataFrame
) -> pd.DataFrame:
    subscription_features = build_subscription_features(subscriptions)
    support_features = build_support_features(support_tickets)
    usage_features = build_feature_usage_features(feature_usage, subscriptions)


    account_level_features = (
        accounts
        .merge(subscription_features, on="account_id", how="left")
        .merge(usage_features, on="account_id", how="left")
        .merge(support_features, on="account_id", how="left")
    )

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
    account_level_features[flag_columns] = account_level_features[flag_columns].fillna(False).astype(bool)

    return account_level_features