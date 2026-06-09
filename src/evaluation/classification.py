import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_binary_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Binary Classifier",
    threshold: float = 0.5,
) -> dict[str, object]:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    classification_report_text = classification_report(y_test, y_pred, zero_division=0)
    classification_report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    metrics: dict[str, object] = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report_dict,
    }

    print(model_name)
    print("=" * 80)
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.3f}")
    print(f"PR AUC:    {metrics['average_precision']:.3f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(classification_report_text)

    return metrics
