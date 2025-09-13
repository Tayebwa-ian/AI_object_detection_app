"""
ResNet backbone wrapper.

Provides:
- feature extraction for images
- prototype creation and prototype-based classification (cosine similarity)
- helpers to produce metrics for classification tasks
- returns inference metadata (inference_time and optionally avg_confidence)
"""
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics.pairwise import cosine_similarity

from ..config import RESNET_TYPE, DEVICE, IMAGE_SIZE
from ..utils.timing import measure_time
from ..utils.metrics import classification_metrics, per_label_metrics

class ResNetWrapper:
    """
    ResNet backbone wrapper for feature extraction and prototype-based classification.

    Note: the backbone is created with an Identity final layer (no classifier). Use few_shot module
    to train a supervised classifier on top of extracted features.
    """

    def __init__(self, model_name: str = None, pretrained: bool = True, device: str = None):
        """
        Initialize backbone and preprocessing pipeline.

        Args:
            model_name: torchvision name (e.g. 'resnet50'). If None uses config.RESNET_TYPE.
            pretrained: whether to use ImageNet pretrained weights.
            device: device string (default: config.DEVICE).
        """
        self.device = device or DEVICE
        self.model_name = model_name or RESNET_TYPE
        self.model = getattr(models, self.model_name)(pretrained=pretrained)

        # Replace final fully-connected with Identity so forward returns features
        # For ResNet this is `fc`
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        # transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # prototypes: maps label -> averaged feature vector (numpy)
        self.prototypes = defaultdict(list)

    def preprocess(self, image) -> torch.Tensor:
        """
        Convert PIL image -> tensor, normalized, on device.
        Returns a minibatch of size 1.
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        return x

    @measure_time
    def extract_features(self, image) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract a feature vector for the given image using ResNet backbone.

        Returns:
            features (np.ndarray): flattened 1D feature vector
            metadata (dict): placeholder metadata; inference_time is added by decorator
        """
        x = self.preprocess(image)
        with torch.no_grad():
            feats = self.model(x)  # shape [1, D]
        feats_np = feats.cpu().numpy().reshape(-1)
        # We return features and allow metadata to be filled by decorator
        return feats_np, {}

    def add_prototype(self, image, label: str):
        """
        Extract features and queue them under the provided label for later prototype creation.

        Args:
            image: PIL image
            label: label name
        """
        features, _ = self.extract_features(image)
        self.prototypes[label].append(features)

    def finalize_prototypes(self):
        """
        Average stored features for each label to produce a single prototype vector per label.
        Converts self.prototypes[label] from list->np.ndarray mean.
        """
        for label, feats in list(self.prototypes.items()):
            if len(feats) == 0:
                continue
            self.prototypes[label] = np.mean(feats, axis=0)

    @measure_time
    def classify_with_prototypes(self, image) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Classify image by cosine similarity to each label prototype.

        Returns:
            result: dict with keys 'predicted_label', 'similarity_scores' (mapping label->score)
            metadata: dict with avg_confidence (max similarity), inference_time added by decorator
        """
        # Ensure prototypes exist
        if not self.prototypes:
            raise ValueError("No prototypes available. Call add_prototype() and finalize_prototypes().")

        features, _ = self.extract_features(image)  # this call already returns a metadata entry but decorator will combine
        labels = list(self.prototypes.keys())
        protos = np.stack([self.prototypes[l] for l in labels], axis=0)
        sims = cosine_similarity([features], protos)[0]  # 1 x N -> vector
        best_idx = int(sims.argmax())
        predicted = labels[best_idx]
        result = {
            "predicted_label": predicted,
            "similarity_scores": {label: float(score) for label, score in zip(labels, sims)}
        }
        metadata = {"avg_confidence": float(sims.max())}
        return result, metadata

    def evaluate_predictions(self, y_true: List, y_pred: List) -> Dict[str, Any]:
        """
        Compute overall and per-label metrics for a set of predictions.

        Returns:
            dict containing 'overall' (accuracy/precision/recall/f1) and 'per_label' metrics
        """
        overall = classification_metrics(y_true, y_pred, average="macro")
        per_label = per_label_metrics(y_true, y_pred)
        return {"overall": overall, "per_label": per_label}
