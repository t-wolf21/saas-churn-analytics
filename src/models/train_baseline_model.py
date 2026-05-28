from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import (
    load_accounts,
    load_feature_usage,
    load_subscriptions,
    load_support_tickets,
)
from src.evaluation.classification import evaluate_binary_classifier
from src.features.account_level_features import build_account_level_features


MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "baseline_logistic_regression.joblib"
TARGET_COLUMN = "churn_flag"
RANDOM_STATE = 42


def load_training_data() -> pd.DataFrame:
    accounts = load_accounts()
    subscriptions = load_subscriptions()
    support_tickets = load_support_tickets()
    feature_usage = load_feature_usage()

    return build_account_level_features(
        accounts=accounts,
        subscriptions=subscriptions,
        support_tickets=support_tickets,
        feature_usage=feature_usage,
    )


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns_to_drop = [
        TARGET_COLUMN,
        "account_id",
        "account_name",
        "signup_date",
    ]

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_model(X: pd.DataFrame) -> Pipeline:
    categorical_columns = [
        column
        for column in X.columns
        if isinstance(X[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(X[column])
        or pd.api.types.is_string_dtype(X[column])
    ]
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    boolean_columns = X.select_dtypes(include=["bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            ("boolean", "passthrough", boolean_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_training_data()
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)

    evaluate_binary_classifier(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="Baseline Logistic Regression",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
