"""
classifiers.py

Classifier utilities for few-shot:
 - Cosine-prototype builder and classifier
 - Logistic Regression trainer & predictor (sklearn)
 - Linear probe trainer & predictor (PyTorch single linear layer)

All functions are implemented to return simple, serializable metadata useful for logging.
"""
from typing import Dict, Any, Tuple, List, Optional
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Cosine prototype utilities
# -------------------------------
def build_prototypes_from_samples(X: np.ndarray, y: List[str]) -> Dict[str, np.ndarray]:
    """
    Given feature matrix X (N,D) and labels y (N,), compute L2-normalized prototype (mean) per label.

    Returns mapping label -> prototype (D,)
    """
    prototypes = {}
    unique_labels = sorted(set(y))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(y) if l == lab]
        if not idx:
            continue
        mat = X[idx]
        proto = np.mean(mat, axis=0)
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        prototypes[lab] = proto.astype(np.float32)
    return prototypes

def classify_with_prototypes(feature: np.ndarray, prototypes: Dict[str, np.ndarray], top_k: int = 1) -> List[Tuple[str, float]]:
    """
    Cosine similarity classification of single feature against prototypes.
    Returns list of (label, score) sorted desc.
    """
    if not prototypes:
        return []
    labels = list(prototypes.keys())
    proto_mat = np.vstack([prototypes[l] for l in labels])
    f = feature.astype(np.float32)
    if np.linalg.norm(f) > 0:
        f = f / np.linalg.norm(f)
    sims = proto_mat.dot(f)  # (L,)
    order = np.argsort(-sims)
    return [(labels[i], float(sims[i])) for i in order[:top_k]]

# -------------------------------
# Logistic regression wrappers
# -------------------------------
def train_logistic_regression(X: np.ndarray, y: List[str], **lr_kwargs) -> Dict[str, Any]:
    """
    Train a logistic regression classifier on features.
    Returns a dict with fitted classifier, label encoder, and metadata.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = LogisticRegression(**lr_kwargs)
    t0 = time.time()
    clf.fit(X, y_enc)
    elapsed = time.time() - t0
    meta = {"model_type": "logistic_regression", "train_time": elapsed, "n_samples": int(X.shape[0])}
    return {"classifier": clf, "label_encoder": le, "meta": meta}

def predict_with_logistic(clf_obj: Dict[str, Any], X: np.ndarray) -> List[Tuple[str, float]]:
    """
    Predict label and score for a single feature X (1,D) using fitted logistic regression object.
    Returns list [(label, prob)] with single top result.
    """
    clf = clf_obj["classifier"]
    le = clf_obj["label_encoder"]
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X.reshape(1, -1))[0]
        top_idx = int(np.argmax(probs))
        return [(le.inverse_transform([top_idx])[0], float(probs[top_idx]))]
    else:
        pred_idx = int(clf.predict(X.reshape(1, -1))[0])
        return [(le.inverse_transform([pred_idx])[0], 1.0)]

# -------------------------------
# Linear probe (PyTorch)
# -------------------------------
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_linear_probe(X: np.ndarray, y: List[str], epochs: int = 30, lr: float = 1e-2, batch_size: int = 32,
                       weight_path: Optional[str] = None, device: str = "cpu") -> Dict[str, Any]:
    """
    Train a linear probe (single linear layer) on frozen features using PyTorch.
    Returns dict with saved path (if provided), label encoder, model state, and metadata.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    input_dim = X.shape[1]
    model = LinearProbe(input_dim, num_classes).to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(device)

    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    n = X.shape[0]
    steps_per_epoch = max(1, n // batch_size)

    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        # simple random shuffle batching
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    elapsed = time.time() - t0

    # Save if requested
    saved_path = None
    if weight_path:
        os.makedirs(os.path.dirname(weight_path) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "num_classes": num_classes}, weight_path)
        saved_path = os.path.abspath(weight_path)

    meta = {"model_type": "linear_probe", "train_time": elapsed, "n_samples": int(n), "epochs": int(epochs)}
    return {"model": model.eval(), "label_encoder": le, "meta": meta, "weight_path": saved_path}

def predict_with_linear_probe(probe_obj: Dict[str, Any], feature: np.ndarray, device: str = "cpu") -> Tuple[str, float]:
    """
    Predict with trained linear probe object (returned by train_linear_probe).
    Returns (label, score) for top class (softmax prob).
    """
    model = probe_obj["model"]
    le = probe_obj["label_encoder"]
    model = model.to(device)
    import torch.nn.functional as F
    x = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
    top_idx = int(np.argmax(probs))
    return le.inverse_transform([top_idx])[0], float(probs[top_idx])
