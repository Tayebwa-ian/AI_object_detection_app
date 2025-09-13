"""
Few-shot training utilities.

Implements a simple logistic regression classifier trained on features extracted
from ResNetWrapper. Designed for few-shot/low-data scenarios.

Returns trained classifier and provides evaluation helpers.
"""
from typing import List, Tuple, Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from ..utils.metrics import classification_metrics, per_label_metrics

class FewShotClassifier:
    """
    Few-shot classifier that trains a LogisticRegression on extracted features.

    Typical usage:
        fs = FewShotClassifier()
        fs.fit(X_train_features, y_train_labels)
        preds = fs.predict(X_test_features)
        metrics = fs.evaluate(y_test, preds)
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 0):
        """
        Args:
            C: inverse regularization strength for LogisticRegression
            max_iter: maximum iterations
        """
        self.clf = LogisticRegression(C=C, max_iter=max_iter, multi_class='multinomial', solver='lbfgs')
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X: List[np.ndarray], y: List[str]):
        """
        Fit classifier on feature vectors.

        Args:
            X: list/array (N, D) of features
            y: list of N labels (strings)
        """
        X_arr = np.stack(X, axis=0)
        y_enc = self.label_encoder.fit_transform(y)
        self.clf.fit(X_arr, y_enc)
        self.is_fitted = True

    def predict(self, X: List[np.ndarray]) -> List[str]:
        """
        Predict labels for feature vectors.

        Returns:
            list of label strings
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() before predict().")
        X_arr = np.stack(X, axis=0)
        y_pred_enc = self.clf.predict(X_arr)
        return list(self.label_encoder.inverse_transform(y_pred_enc))

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted.")
        return self.clf.predict_proba(np.stack(X, axis=0))

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        """
        overall = classification_metrics(y_true, y_pred, average="macro")
        per_label = per_label_metrics(y_true, y_pred)
        return {"overall": overall, "per_label": per_label}
