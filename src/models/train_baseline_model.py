from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import (
    load_accounts,
    load_churn_events,
    load_feature_usage,
    load_subscriptions,
    load_support_tickets,
)
from src.evaluation.classification import evaluate_binary_classifier
from src.features.account_level_features import build_account_level_features


MODEL_DIR = Path("data/models")
MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}
TARGET_COLUMN = "churn_flag"
RANDOM_STATE = 42


def load_training_data() -> pd.DataFrame:
    accounts = load_accounts()
    subscriptions = load_subscriptions()
    support_tickets = load_support_tickets()
    feature_usage = load_feature_usage()
    churn_events = load_churn_events()

    return build_account_level_features(
        accounts=accounts,
        subscriptions=subscriptions,
        support_tickets=support_tickets,
        feature_usage=feature_usage,
        churn_events=churn_events,
    )


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns_to_drop = [
        TARGET_COLUMN,
        "account_id",
        "account_name",
        "signup_date",
        "snapshot_date",
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
    return preprocessor


def build_random_forest_model(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]
    )

def build_logistic_regression_model(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
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

    models = {
        "Logistic Regression": build_logistic_regression_model(X_train),
        "Random Forest": build_random_forest_model(X_train),
    }

    thresholds = [0.3, 0.4, 0.5]

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        for threshold in thresholds:
            evaluate_binary_classifier(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=f"{model_name} threshold={threshold}",
                threshold=threshold,
            )

        joblib.dump(model, MODEL_PATHS[model_name])


if __name__ == "__main__":
    main()
