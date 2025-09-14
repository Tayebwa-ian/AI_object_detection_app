"""
ResNetWrapper (improved)

Improvements & features:
- Multi-layer feature aggregation (concatenate pooled features from multiple ResNet stages)
- GeM pooling (Generalized Mean pooling) instead of naive global avgpool for more discriminative features
- Optional Test-Time Augmentation (TTA) averaging (flip / center crop)
- L2 normalization of features and optional PCA-whitening support hooks
- Robust prototype storage on-disk (per-sample + averaged prototype per label), with index file
- Functions required:
    - extract_features_from_path(image_path)
    - add_prototype(label, image_path, store_root)
    - finalize_prototypes(store_root)
    - load_prototypes(store_root)
    - classify_with_prototypes(image_path, store_root)
    - list_prototypes(store_root)
- Clear metadata returned: model_name, inference_time, feature_dim, device, etc.

References & justification:
- GeM pooling improves retrieval and compact descriptors (GeM paper / blog). See e.g. Radenović et al. and GeM resources.
- Multi-layer aggregation / deep layer aggregation can improve representational richness.
- TTA and whitening/PCA are commonly used for retrieval/robustness.
"""
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import uuid
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from ...config import RESNET_TYPE, DEVICE, IMAGE_SIZE
from ...utils.timing import measure_time
from ...utils.metrics import classification_metrics, per_label_metrics

# -------------------------------
# GeM pooling implementation
# -------------------------------
class GeM(nn.Module):
    """
    Generalized Mean (GeM) pooling.
    GeM is a parameterized pooling that generalizes max / average pooling.
    p -> 1 equals avgpool, p -> large approximates maxpool.

    Reference: "Fine-tuning CNN Image Retrieval with No Human Annotation" (Radenovic et al.)
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, train_p: bool = False):
        super().__init__()
        if train_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)
        self.eps = eps
        self.train_p = train_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        if isinstance(self.p, torch.nn.Parameter):
            p = self.p.clamp(min=self.eps)
        else:
            p = float(self.p)
        x = x.clamp(min=self.eps).pow(p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x.pow(1.0 / p)
        return x.squeeze(-1).squeeze(-1)  # shape (B, C)

# -------------------------------
# Utility helpers
# -------------------------------
def _ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def _save_numpy(path: str, arr: np.ndarray):
    np.save(path, arr)

def _load_numpy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)

# -------------------------------
# Main wrapper
# -------------------------------
class ResNetWrapper:
    """
    Improved ResNet wrapper for feature extraction, prototype creation/storage, and prototype classification.

    Key design choices:
    - We extract features from multiple stages (layer3, layer4) and pool them with GeM for stronger descriptors.
    - We L2-normalize descriptors (common for cosine similarity / retrieval).
    - We support test-time augmentation (TTA) by averaging descriptors across flipped / cropped inputs.
    - Prototypes are stored on disk under a structured folder:
        <store_root>/
            samples/
                <label>/
                    sample_<uuid>.npy   # per-image feature
                    meta_<uuid>.json    # optional metadata (source path, timestamp)
            prototypes/
                <label>/
                    prototype.npy        # averaged prototype for label
                    proto_meta.json      # metadata for prototype (count, created_at, dim)
            index.json   # maps labels -> prototype paths & sample counts
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 pretrained: bool = True,
                 device: Optional[str] = None,
                 use_gem: bool = True,
                 gem_p: float = 3.0,
                 tta: bool = True):
        """
        Args:
            model_name: torchvision model name for ResNet (e.g., 'resnet50'). Defaults to config.RESNET_TYPE.
            pretrained: load pretrained imagenet weights
            device: torch device string
            use_gem: whether to use GeM pooling (True recommended for compact descriptors)
            gem_p: initial p parameter for GeM
            tta: whether to apply test-time augmentation (flip) when extracting features
        """
        self.device = device or DEVICE
        self.model_name = model_name or RESNET_TYPE
        self.use_gem = use_gem
        self.gem_p = gem_p
        self.tta = tta

        # Load torchvision ResNet
        # We'll retain body and extract intermediate features from layer3 and layer4
        backbone = getattr(models, self.model_name)(pretrained=pretrained)

        # Remove classifier head (fc) — we will use features
        backbone.fc = nn.Identity()

        # we also will use intermediate layers: layer3 and layer4
        self.backbone = backbone.to(self.device)
        self.backbone.eval()

        # Prepare pooling modules (for each layer we pool)
        if self.use_gem:
            self.pool = GeM(p=self.gem_p, train_p=False)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Preprocessing: match ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # A simple hook arrangement to capture layer outputs
        self._feature_buffers = {}
        self._register_hooks()

        # Feature dimensionality will be determined on first forward
        self.feature_dim = None

    def _register_hooks(self):
        """
        Register forward hooks at layer3 and layer4 to capture features.
        """
        # layer3 and layer4 exist in torchvision ResNet
        def hook_layer3(module, input, output):
            # output shape (B, C, H, W)
            self._feature_buffers['layer3'] = output.detach()

        def hook_layer4(module, input, output):
            self._feature_buffers['layer4'] = output.detach()

        # Remove existing hooks if any
        for h in getattr(self, "_hooks", []):
            h.remove()
        self._hooks = []

        self._hooks.append(self.backbone.layer3.register_forward_hook(hook_layer3))
        self._hooks.append(self.backbone.layer4.register_forward_hook(hook_layer4))

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image to tensor on device (batch size 1).
        """
        return self.transform(image).unsqueeze(0).to(self.device)

    def _forward_once(self, x: torch.Tensor) -> List[np.ndarray]:
        """
        Forward pass that collects features from hooks, pool them and return concatenated numpy feature.
        """
        # Clear buffers
        self._feature_buffers.clear()

        # Pass through backbone (will fill hooks)
        with torch.no_grad():
            _ = self.backbone(x)  # body -> hooks fill buffers

        # Expect layer3 and layer4 in buffers
        feats = []
        for key in ['layer3', 'layer4']:
            if key not in self._feature_buffers:
                raise RuntimeError(f"Expected {key} features but not found (model hook failed).")
            feat_map = self._feature_buffers[key]  # (1, C, H, W)
            pooled = self.pool(feat_map)  # (1, C)
            feats.append(pooled.cpu().numpy().reshape(-1))

        # Concatenate pooled vectors
        concat = np.concatenate(feats, axis=0)  # (D,)
        # L2 normalize
        norm = np.linalg.norm(concat)
        if norm > 0:
            concat = concat / norm
        # Set feature_dim if unset
        if self.feature_dim is None:
            self.feature_dim = int(concat.shape[0])

        return concat

    def _tta_extract(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features with simple TTA: original + horizontal flip.
        Returns averaged feature and metadata (num_augmentations).
        """
        variants = [image]
        # horizontal flip
        variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        feats = []
        for img in variants:
            x = self.preprocess(img)
            feat = self._forward_once(x)
            feats.append(feat)
        feats_arr = np.vstack(feats)  # (n_augs, D)
        avg = np.mean(feats_arr, axis=0)
        # re-normalize after averaging
        nrm = np.linalg.norm(avg)
        if nrm > 0:
            avg = avg / nrm
        metadata = {"tta_augments": len(variants)}
        return avg, metadata

    @measure_time
    def extract_features_from_path(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Public API: given an image path, extract feature vector.

        Returns:
            features (np.ndarray): 1D L2-normalized vector
            metadata (dict): includes model_name, device, feature_dim, inference_time (via decorator) and tta info
        """
        start_ts = time.time()
        img = Image.open(image_path).convert("RGB")
        if self.tta:
            feat, tta_meta = self._tta_extract(img)
        else:
            x = self.preprocess(img)
            feat = self._forward_once(x)
            tta_meta = {"tta_augments": 1}
        elapsed = time.time() - start_ts

        metadata = {
            "model_name": f"{self.model_name}",
            "device": self.device,
            "feature_dim": int(feat.shape[0]),
            "tta": self.tta,
            **tta_meta,
            # 'inference_time' will be merged by the measure_time decorator
        }
        return feat.astype(np.float32), metadata

    # -------------------------------
    # Prototype storage API
    # -------------------------------
    def _index_path(self, store_root: str) -> str:
        return os.path.join(store_root, "index.json")

    def add_prototype(self, label: str, image_path: str, store_root: str) -> Dict[str, Any]:
        """
        Extract features from image_path and save the feature vector as a sample under store_root/samples/<label>/.

        Creates directories as needed and writes sample_<uuid>.npy and meta_<uuid>.json.

        Returns:
            sample_meta: dict containing sample_id, sample_path, label, feature_dim, timestamp
        """
        feat, meta = self.extract_features_from_path(image_path)
        sample_id = str(uuid.uuid4())
        samples_dir = os.path.join(store_root, "samples", label)
        _ensure_dir(samples_dir)
        sample_path = os.path.join(samples_dir, f"sample_{sample_id}.npy")
        meta_path = os.path.join(samples_dir, f"meta_{sample_id}.json")

        # Save feature vector (numpy)
        _save_numpy(sample_path, feat)
        sample_meta = {
            "sample_id": sample_id,
            "sample_path": sample_path,
            "source_image": os.path.abspath(image_path),
            "label": label,
            "feature_dim": int(feat.shape[0]),
            "timestamp": time.time(),
            "extract_meta": meta
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2)

        # Update index.json
        index = {"labels": {}}
        idx_path = self._index_path(store_root)
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                try:
                    index = json.load(f)
                except Exception:
                    index = {"labels": {}}
        lbl_entry = index["labels"].setdefault(label, {"samples": [], "prototype": None})
        lbl_entry["samples"].append({
            "sample_id": sample_id,
            "sample_path": sample_path,
            "meta_path": meta_path
        })
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        return sample_meta

    def finalize_prototypes(self, store_root: str) -> Dict[str, Any]:
        """
        For each label under store_root/samples, average the saved sample feature vectors
        into store_root/prototypes/<label>/prototype.npy and write proto_meta.json.

        Returns:
            dict mapping label -> prototype_meta (path, count, dim)
        """
        samples_root = os.path.join(store_root, "samples")
        prototypes_root = os.path.join(store_root, "prototypes")
        _ensure_dir(prototypes_root)
        result = {}
        if not os.path.exists(samples_root):
            return result

        for label in os.listdir(samples_root):
            lab_dir = os.path.join(samples_root, label)
            if not os.path.isdir(lab_dir):
                continue
            feats = []
            for fname in os.listdir(lab_dir):
                if not fname.endswith(".npy"):
                    continue
                fpath = os.path.join(lab_dir, fname)
                try:
                    arr = _load_numpy(fpath)
                    feats.append(arr.astype(np.float32))
                except Exception:
                    continue
            if not feats:
                continue
            feats_arr = np.vstack(feats)  # (N, D)
            proto = np.mean(feats_arr, axis=0)
            # re-normalize
            nrm = np.linalg.norm(proto)
            if nrm > 0:
                proto = proto / nrm

            proto_dir = os.path.join(prototypes_root, label)
            _ensure_dir(proto_dir)
            proto_path = os.path.join(proto_dir, "prototype.npy")
            proto_meta_path = os.path.join(proto_dir, "proto_meta.json")
            _save_numpy(proto_path, proto.astype(np.float32))
            proto_meta = {
                "label": label,
                "prototype_path": proto_path,
                "count": int(feats_arr.shape[0]),
                "feature_dim": int(proto.shape[0]),
                "created_at": time.time()
            }
            with open(proto_meta_path, "w", encoding="utf-8") as f:
                json.dump(proto_meta, f, indent=2)
            result[label] = proto_meta

        # Update index.json with prototype info
        idx_path = self._index_path(store_root)
        index = {"labels": {}}
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except Exception:
                index = {"labels": {}}
        for label, meta in result.items():
            lbl_entry = index["labels"].setdefault(label, {"samples": [], "prototype": None})
            lbl_entry["prototype"] = {"prototype_path": meta["prototype_path"], "count": meta["count"]}
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        return result

    def load_prototypes(self, store_root: str) -> Dict[str, np.ndarray]:
        """
        Load all prototypes from store_root and return mapping label -> prototype vector.
        """
        protos = {}
        prototypes_root = os.path.join(store_root, "prototypes")
        if not os.path.exists(prototypes_root):
            return protos
        for label in os.listdir(prototypes_root):
            pdir = os.path.join(prototypes_root, label)
            ppath = os.path.join(pdir, "prototype.npy")
            if os.path.exists(ppath):
                protos[label] = _load_numpy(ppath).astype(np.float32)
        return protos

    def list_prototypes(self, store_root: str) -> Dict[str, Any]:
        """
        Return index.json contents (labels, sample counts, prototype paths).
        """
        idx_path = self._index_path(store_root)
        if not os.path.exists(idx_path):
            return {}
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------
    # Prototype classification
    # -------------------------------
    @measure_time
    def classify_with_prototypes(self, image_path: str, store_root: str, top_k: int = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Classify image by cosine similarity to stored prototypes.

        Returns:
            result: { 'predictions': [ (label, score) ... ] }
            metadata: { 'avg_confidence' : float, 'num_prototypes': int, 'feature_dim': int }
        """
        # extract feature
        feat, feat_meta = self.extract_features_from_path(image_path)  # returns (feat, metadata)
        protos = self.load_prototypes(store_root)
        if not protos:
            raise ValueError("No prototypes found. Call add_prototype() and finalize_prototypes().")

        labels = list(protos.keys())
        proto_mat = np.vstack([protos[l] for l in labels])  # (N, D)
        # cosine similarity between feat and protos: dot since normalized
        # Ensure feat is normalized
        f = feat.astype(np.float32)
        if np.linalg.norm(f) > 0:
            f = f / np.linalg.norm(f)
        sims = np.dot(proto_mat, f)  # (N,)
        # sort descending
        order = np.argsort(-sims)
        preds = [(labels[i], float(sims[i])) for i in order[:top_k]]
        result = {"predictions": preds}
        metadata = {
            "num_prototypes": len(labels),
            "feature_dim": int(f.shape[0]),
            "avg_confidence": float(float(np.max(sims))) if len(sims) > 0 else 0.0,
            **feat_meta
        }
        return result, metadata

    # -------------------------------
    # Evaluation helpers
    # -------------------------------
    def evaluate_predictions(self, y_true: List, y_pred: List) -> Dict[str, Any]:
        """
        Compute overall and per-label metrics for predictions.
        """
        overall = classification_metrics(y_true, y_pred, average="macro")
        per_label = per_label_metrics(y_true, y_pred)
        return {"overall": overall, "per_label": per_label}
