"""
Evaluation metrics utilities.

Provides classification metrics: accuracy, precision, recall, F1.
Also provides per-label metrics (precision/recall/F1) using sklearn.
"""
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def classification_metrics(y_true: List, y_pred: List, average: str = "macro") -> Dict[str, float]:
    """
    Compute overall classification metrics.

    Args:
        y_true: ground-truth labels (list-like)
        y_pred: predicted labels (list-like)
        average: averaging strategy for precision/recall/f1 (default "macro")

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }

def per_label_metrics(y_true: List, y_pred: List) -> Dict[str, Dict[str, float]]:
    """
    Compute per-label precision, recall, f1 and support via sklearn's classification_report.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels

    Returns:
        dict mapping label -> {precision, recall, f1, support}
    """
    # classification_report returns a string or dict if output_dict=True
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # report contains per-label entries and 'macro avg'/'weighted avg'
    # convert numpy types to floats for JSON friendliness
    cleaned = {}
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            cleaned[label] = {
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1-score", metrics.get("f1", 0.0))),
                "support": int(metrics.get("support", 0))
            }
    return cleaned
