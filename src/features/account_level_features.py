import pandas as pd

def build_subscription_features(subscriptions : pd.DataFrame) -> pd.DataFrame:
    subscriptions = subscriptions.copy()
    subscriptions = subscriptions.groupby("account_id").agg(
        avg_mrr=("mrr_amount", "mean"),
        max_mrr=("mrr_amount", "max"),
        total_mrr=("mrr_amount", "sum"),
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

def build_support_features(support_tickets : pd.DataFrame) -> pd.DataFrame:
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

def build_feature_usage_features(feature_usage : pd.DataFrame) -> pd.DataFrame:
    feature_usage = feature_usage.copy()
    feature_usage = feature_usage.groupby("subscription_id").agg(
        total_usage_count=("usage_count", "sum"),
        avg_usage_count=("usage_count", "mean"),
        total_usage_duration_secs=("usage_duration_secs", "sum"),
        avg_usage_duration_secs=("usage_duration_secs", "mean"),
        unique_features_used=("feature_name", "nunique"),
        total_error_count=("error_count", "sum"),
        avg_error_count=("error_count", "mean"),
        beta_feature_rate=("is_beta_feature", "mean"),
        has_used_feature=("feature_name", "any"),
        usage_event_count=("usage_id", "count")
    ).reset_index()
    return feature_usage