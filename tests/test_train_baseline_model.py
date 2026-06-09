import numpy as np
import pandas as pd

from src.models import train_baseline_model as tbm


def test_model_dir_is_anchored_to_project_root():
    assert tbm.MODEL_DIR == tbm.PROJECT_ROOT / "data" / "models"


def test_split_features_and_target_drops_metadata_columns_and_returns_int_target():
    df = pd.DataFrame(
        {
            "account_id": [101, 102],
            "account_name": ["Acme", "Globex"],
            "signup_date": ["2024-01-01", "2024-01-02"],
            "snapshot_date": ["2024-03-01", "2024-03-02"],
            "churn_flag": [True, False],
            "feature_a": [1.5, 2.5],
            "feature_b": ["x", "y"],
        }
    )

    X, y = tbm.split_features_and_target(df)

    assert list(X.columns) == ["feature_a", "feature_b"]
    pd.testing.assert_series_equal(
        y,
        pd.Series([1, 0], name="churn_flag", dtype="int64"),
    )


def test_fixed_threshold_is_default_value():
    assert tbm.FIXED_THRESHOLD == 0.35


def test_split_train_validation_test_creates_expected_partition_sizes():
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series([0, 1] * 5, name="churn_flag")

    X_train, X_val, X_test, y_train, y_val, y_test = tbm.split_train_validation_test(X, y)

    assert len(X_train) == 6
    assert len(X_val) == 2
    assert len(X_test) == 2
    assert y_train.value_counts().to_dict() == {0: 3, 1: 3}
    assert y_val.value_counts().to_dict() == {0: 1, 1: 1}
    assert y_test.value_counts().to_dict() == {0: 1, 1: 1}


def test_build_preprocessor_separates_column_types():
    X = pd.DataFrame(
        {
            "numeric_feature": [1.0, np.nan, 3.0],
            "categorical_feature": ["a", "b", "a"],
            "string_feature": ["foo", "bar", "foo"],
            "boolean_feature": [True, False, True],
        }
    )

    preprocessor = tbm.build_preprocessor(X)
    transformers = {name: columns for name, _, columns in preprocessor.transformers}

    assert transformers["numeric"] == ["numeric_feature"]
    assert transformers["categorical"] == ["categorical_feature", "string_feature"]
    assert transformers["boolean"] == ["boolean_feature"]

    transformed = preprocessor.fit_transform(X)
    assert transformed.shape == (3, 6)


def test_evaluate_binary_classifier_returns_structured_metrics(capsys):
    class DummyModel:
        def predict_proba(self, X):
            return np.array(
                [
                    [0.10, 0.90],
                    [0.80, 0.20],
                    [0.30, 0.70],
                    [0.90, 0.10],
                ]
            )

    metrics = tbm.evaluate_binary_classifier(
        model=DummyModel(),
        X_test=pd.DataFrame({"feature": [1, 2, 3, 4]}),
        y_test=pd.Series([1, 0, 1, 0]),
        model_name="Dummy Model",
        threshold=0.5,
    )

    captured = capsys.readouterr().out
    assert "PR AUC:" in captured
    assert metrics["threshold"] == 0.5
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    np.testing.assert_array_equal(metrics["confusion_matrix"], np.array([[2, 0], [0, 2]]))
    assert metrics["classification_report"]["accuracy"] == 1.0


def test_main_uses_validation_split_and_saves_artifacts(monkeypatch):
    training_df = pd.DataFrame(
        {
            "account_id": list(range(10)),
            "account_name": [f"account-{idx}" for idx in range(10)],
            "signup_date": ["2024-01-01"] * 10,
            "snapshot_date": ["2024-02-01"] * 10,
            "churn_flag": [0, 1] * 5,
            "numeric_feature": np.linspace(1.0, 10.0, 10),
            "categorical_feature": ["x", "y"] * 5,
            "boolean_feature": [True, False] * 5,
        }
    )

    evaluation_calls = []
    dump_calls = []
    built_models = []

    class DummyModel:
        def __init__(self, name: str):
            self.name = name
            self.fit_calls = 0
            self.fitted_shapes = []

        def fit(self, X, y):
            self.fit_calls += 1
            self.fitted_shapes.append((X.shape, y.shape))
            return self

    def fake_load_training_data():
        return training_df

    def fake_build_logistic_regression_model(X):
        model = DummyModel("Logistic Regression")
        built_models.append(model)
        return model

    def fake_build_random_forest_model(X):
        model = DummyModel("Random Forest")
        built_models.append(model)
        return model

    def fake_evaluate_binary_classifier(model, X_test, y_test, model_name, threshold):
        score = 0.80 if "validation" in model_name else 0.90
        metrics = {
            "threshold": threshold,
            "accuracy": score,
            "precision": score,
            "recall": score,
            "f1": score,
            "roc_auc": 0.5,
            "average_precision": 0.5,
            "confusion_matrix": np.array([[1, 0], [0, 1]]),
            "classification_report": {"accuracy": score},
        }
        evaluation_calls.append(
            {
                "model": model.name,
                "model_name": model_name,
                "threshold": threshold,
                "test_rows": len(X_test),
                "returned_f1": metrics["f1"],
            }
        )
        return metrics

    def fake_split_train_validation_test(X, y):
        X_train = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6]})
        X_val = pd.DataFrame({"feature": [7, 8]})
        X_test = pd.DataFrame({"feature": [9, 10]})
        y_train = pd.Series([0, 1, 0, 1, 0, 1], name="churn_flag")
        y_val = pd.Series([0, 1], name="churn_flag")
        y_test = pd.Series([0, 1], name="churn_flag")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fake_joblib_dump(obj, path):
        dump_calls.append((obj, path))

    monkeypatch.setattr(tbm, "load_training_data", fake_load_training_data)
    monkeypatch.setattr(tbm, "split_train_validation_test", fake_split_train_validation_test)
    monkeypatch.setattr(tbm, "build_logistic_regression_model", fake_build_logistic_regression_model)
    monkeypatch.setattr(tbm, "build_random_forest_model", fake_build_random_forest_model)
    monkeypatch.setattr(tbm, "evaluate_binary_classifier", fake_evaluate_binary_classifier)
    monkeypatch.setattr(tbm.joblib, "dump", fake_joblib_dump)

    tbm.main()

    assert [model.fit_calls for model in built_models] == [2, 2]
    assert [call["model_name"] for call in evaluation_calls] == [
        "Logistic Regression validation threshold=0.35",
        "Logistic Regression test threshold=0.35",
        "Random Forest validation threshold=0.35",
        "Random Forest test threshold=0.35",
    ]
    assert [call["threshold"] for call in evaluation_calls] == [0.35, 0.35, 0.35, 0.35]
    assert [call["test_rows"] for call in evaluation_calls] == [2, 2, 2, 2]

    assert dump_calls[0][1] == tbm.MODEL_PATHS["Logistic Regression"]
    assert dump_calls[1][1] == tbm.MODEL_PATHS["Random Forest"]
    assert dump_calls[0][0]["threshold"] == 0.35
    assert dump_calls[1][0]["threshold"] == 0.35
    assert dump_calls[0][0]["validation_metrics"]["accuracy"] == 0.80
    assert dump_calls[1][0]["validation_metrics"]["accuracy"] == 0.80
    assert dump_calls[0][0]["test_metrics"]["accuracy"] == 0.90
    assert dump_calls[1][0]["test_metrics"]["accuracy"] == 0.90
