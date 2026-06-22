import json

import numpy as np
import pandas as pd

from src.models import train_baseline_model as tbm


def test_model_dir_is_anchored_to_project_root():
    assert tbm.MODEL_DIR == tbm.PROJECT_ROOT / "data" / "models"


def test_report_paths_are_anchored_to_project_root():
    assert tbm.REPORT_DIR == tbm.PROJECT_ROOT / "reports"
    assert tbm.METRICS_REPORT_PATH == tbm.REPORT_DIR / "model_metrics.json"
    assert tbm.FEATURE_IMPORTANCE_REPORT_PATH == tbm.REPORT_DIR / "feature_importance.csv"


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


def test_select_best_threshold_returns_threshold_with_highest_validation_f1():
    class DummyModel:
        def predict_proba(self, X):
            return np.array(
                [
                    [0.10, 0.90],
                    [0.60, 0.40],
                    [0.40, 0.60],
                    [0.80, 0.20],
                ]
            )

    threshold = tbm.select_best_threshold(
        model=DummyModel(),
        X_val=pd.DataFrame({"feature": [1, 2, 3, 4]}),
        y_val=pd.Series([1, 0, 1, 0]),
        thresholds=np.array([0.30, 0.50, 0.70]),
    )

    assert threshold == 0.50


def test_write_model_metrics_report_serializes_numpy_values(tmp_path):
    report_path = tmp_path / "model_metrics.json"

    tbm.write_model_metrics_report(
        {
            "Dummy Model": {
                "threshold": np.float64(0.42),
                "validation_metrics": {
                    "f1": np.float64(0.8),
                    "confusion_matrix": np.array([[1, 0], [0, 1]]),
                },
                "test_metrics": {
                    "f1": np.float64(0.9),
                    "confusion_matrix": np.array([[2, 0], [1, 1]]),
                },
            }
        },
        path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["random_state"] == tbm.RANDOM_STATE
    assert report["threshold_selection_metric"] == "f1"
    assert report["models"]["Dummy Model"]["threshold"] == 0.42
    assert report["models"]["Dummy Model"]["validation_metrics"]["confusion_matrix"] == [[1, 0], [0, 1]]


def test_extract_feature_importance_returns_sorted_random_forest_importances():
    X = pd.DataFrame(
        {
            "numeric_feature": [0.0, 1.0, 2.0, 3.0],
            "categorical_feature": ["a", "b", "a", "b"],
            "boolean_feature": [False, True, False, True],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="churn_flag")
    model = tbm.build_random_forest_model(X)
    model.fit(X, y)

    feature_importance = tbm.extract_feature_importance(model, "Random Forest")

    assert list(feature_importance.columns) == ["model", "feature", "importance"]
    assert not feature_importance.empty
    assert feature_importance["model"].eq("Random Forest").all()
    assert feature_importance["importance"].is_monotonic_decreasing


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
    threshold_selection_calls = []
    dump_calls = []
    metrics_report_calls = []
    feature_importance_extraction_calls = []
    feature_importance_report_calls = []
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

    def fake_select_best_threshold(model, X_val, y_val):
        threshold = 0.60 if model.name == "Logistic Regression" else 0.42
        threshold_selection_calls.append(
            {
                "model": model.name,
                "validation_rows": len(X_val),
                "threshold": threshold,
            }
        )
        return threshold

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

    def fake_write_model_metrics_report(model_results):
        metrics_report_calls.append(model_results)

    def fake_extract_feature_importance(model, model_name):
        feature_importance_extraction_calls.append((model.name, model_name))
        if model_name == "Random Forest":
            return pd.DataFrame(
                {
                    "model": [model_name],
                    "feature": ["numeric_feature"],
                    "importance": [1.0],
                }
            )
        return pd.DataFrame(columns=["model", "feature", "importance"])

    def fake_write_feature_importance_report(feature_importance_frames):
        feature_importance_report_calls.append(feature_importance_frames)

    monkeypatch.setattr(tbm, "load_training_data", fake_load_training_data)
    monkeypatch.setattr(tbm, "split_train_validation_test", fake_split_train_validation_test)
    monkeypatch.setattr(tbm, "build_logistic_regression_model", fake_build_logistic_regression_model)
    monkeypatch.setattr(tbm, "build_random_forest_model", fake_build_random_forest_model)
    monkeypatch.setattr(tbm, "select_best_threshold", fake_select_best_threshold)
    monkeypatch.setattr(tbm, "evaluate_binary_classifier", fake_evaluate_binary_classifier)
    monkeypatch.setattr(tbm, "write_model_metrics_report", fake_write_model_metrics_report)
    monkeypatch.setattr(tbm, "extract_feature_importance", fake_extract_feature_importance)
    monkeypatch.setattr(tbm, "write_feature_importance_report", fake_write_feature_importance_report)
    monkeypatch.setattr(tbm.joblib, "dump", fake_joblib_dump)

    tbm.main()

    assert [model.fit_calls for model in built_models] == [2, 2]
    assert threshold_selection_calls == [
        {"model": "Logistic Regression", "validation_rows": 2, "threshold": 0.60},
        {"model": "Random Forest", "validation_rows": 2, "threshold": 0.42},
    ]
    assert [call["model_name"] for call in evaluation_calls] == [
        "Logistic Regression validation threshold=0.60",
        "Logistic Regression test threshold=0.60",
        "Random Forest validation threshold=0.42",
        "Random Forest test threshold=0.42",
    ]
    assert [call["threshold"] for call in evaluation_calls] == [0.6, 0.6, 0.42, 0.42]
    assert [call["test_rows"] for call in evaluation_calls] == [2, 2, 2, 2]

    assert dump_calls[0][1] == tbm.MODEL_PATHS["Logistic Regression"]
    assert dump_calls[1][1] == tbm.MODEL_PATHS["Random Forest"]
    assert dump_calls[0][0]["threshold"] == 0.6
    assert dump_calls[1][0]["threshold"] == 0.42
    assert dump_calls[0][0]["threshold_selection_metric"] == "f1"
    assert dump_calls[1][0]["threshold_selection_metric"] == "f1"
    assert dump_calls[0][0]["validation_metrics"]["accuracy"] == 0.80
    assert dump_calls[1][0]["validation_metrics"]["accuracy"] == 0.80
    assert dump_calls[0][0]["test_metrics"]["accuracy"] == 0.90
    assert dump_calls[1][0]["test_metrics"]["accuracy"] == 0.90

    assert list(metrics_report_calls[0]) == ["Logistic Regression", "Random Forest"]
    assert metrics_report_calls[0]["Logistic Regression"]["threshold"] == 0.6
    assert metrics_report_calls[0]["Random Forest"]["threshold"] == 0.42
    assert feature_importance_extraction_calls == [
        ("Logistic Regression", "Logistic Regression"),
        ("Random Forest", "Random Forest"),
    ]
    assert len(feature_importance_report_calls[0]) == 2


def test_main_runs_end_to_end_with_synthetic_training_data(monkeypatch, tmp_path):
    row_count = 30
    training_df = pd.DataFrame(
        {
            "account_id": range(row_count),
            "account_name": [f"account-{idx}" for idx in range(row_count)],
            "signup_date": ["2024-01-01"] * row_count,
            "snapshot_date": ["2024-03-01"] * row_count,
            "churn_flag": [0, 1] * (row_count // 2),
            "usage_count": [idx * 3 + 10 for idx in range(row_count)],
            "ticket_count": [idx % 5 for idx in range(row_count)],
            "plan_tier": ["Basic", "Pro", "Enterprise"] * 10,
            "has_usage": [idx % 2 == 0 for idx in range(row_count)],
        }
    )

    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    model_paths = {
        "Logistic Regression": model_dir / "logistic_regression.joblib",
        "Random Forest": model_dir / "random_forest.joblib",
    }
    metrics_report_path = report_dir / "model_metrics.json"
    feature_importance_report_path = report_dir / "feature_importance.csv"

    monkeypatch.setattr(tbm, "load_training_data", lambda: training_df)
    monkeypatch.setattr(tbm, "MODEL_DIR", model_dir)
    monkeypatch.setattr(tbm, "REPORT_DIR", report_dir)
    monkeypatch.setattr(tbm, "MODEL_PATHS", model_paths)
    monkeypatch.setattr(tbm, "METRICS_REPORT_PATH", metrics_report_path)
    monkeypatch.setattr(tbm, "FEATURE_IMPORTANCE_REPORT_PATH", feature_importance_report_path)

    tbm.main()

    assert model_paths["Logistic Regression"].exists()
    assert model_paths["Random Forest"].exists()
    assert metrics_report_path.exists()
    assert feature_importance_report_path.exists()

    logistic_artifact = tbm.joblib.load(model_paths["Logistic Regression"])
    random_forest_artifact = tbm.joblib.load(model_paths["Random Forest"])
    assert 0.0 <= logistic_artifact["threshold"] <= 1.0
    assert 0.0 <= random_forest_artifact["threshold"] <= 1.0
    assert logistic_artifact["threshold_selection_metric"] == "f1"
    assert random_forest_artifact["threshold_selection_metric"] == "f1"

    metrics_report = json.loads(metrics_report_path.read_text(encoding="utf-8"))
    assert list(metrics_report["models"]) == ["Logistic Regression", "Random Forest"]
    assert metrics_report["threshold_selection_metric"] == "f1"

    feature_importance_report = pd.read_csv(feature_importance_report_path)
    assert not feature_importance_report.empty
    assert feature_importance_report["model"].eq("Random Forest").all()
    assert set(feature_importance_report.columns) == {"model", "feature", "importance"}
