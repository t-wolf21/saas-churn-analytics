from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import (
    PROJECT_ROOT,
    load_accounts,
    load_churn_events,
    load_feature_usage,
    load_subscriptions,
    load_support_tickets,
)
from src.evaluation.classification import evaluate_binary_classifier
from src.features.account_level_features import build_account_level_features


MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}
TARGET_COLUMN = "churn_flag"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
THRESHOLD_SELECTION_METRIC = "f1"


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


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    holdout_size = VALIDATION_SIZE + TEST_SIZE
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=TEST_SIZE / holdout_size,
        stratify=y_holdout,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = [
        column
        for column in X.columns
        if isinstance(X[column].dtype, pd.CategoricalDtype)
           or pd.api.types.is_object_dtype(X[column])
           or pd.api.types.is_string_dtype(X[column])
    ]
    boolean_columns = X.select_dtypes(include=["bool"]).columns.tolist()
    numeric_columns = [
        column for column in X.select_dtypes(include=["number"]).columns
        if column not in boolean_columns
    ]

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


def build_model_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", classifier),
        ]
    )


def build_random_forest_model(X: pd.DataFrame) -> Pipeline:
    return build_model_pipeline(
        X,
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=1,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def build_logistic_regression_model(X: pd.DataFrame) -> Pipeline:
    return build_model_pipeline(
        X,
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )


def select_best_threshold(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: np.ndarray | None = None,
) -> float:
    if thresholds is None:
        thresholds = THRESHOLD_GRID

    y_proba = model.predict_proba(X_val)[:, 1]
    scores = [
        f1_score(y_val, (y_proba >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index])


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data()
    X, y = split_features_and_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)

    models = {
        "Logistic Regression": build_logistic_regression_model(X_train),
        "Random Forest": build_random_forest_model(X_train),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        threshold = select_best_threshold(model, X_val, y_val)

        validation_metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_val,
            y_test=y_val,
            model_name=f"{model_name} validation threshold={threshold:.2f}",
            threshold=threshold,
        )

        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        model.fit(X_train_full, y_train_full)

        test_metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{model_name} test threshold={threshold:.2f}",
            threshold=threshold,
        )

        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            },
            MODEL_PATHS[model_name],
        )



if __name__ == "__main__":
    main()
