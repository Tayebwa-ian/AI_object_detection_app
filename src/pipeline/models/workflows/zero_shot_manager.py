"""
zero_shot_manager.py

Zero-shot classification manager that works at the segment (object) level.

Capabilities:
- Uses SAMWrapper to segment an image into candidate object regions/segments.
- Supports two zero-shot strategies:
    1) CLIP-based zero-shot (image-text similarity). Requires the `clip` package.
    2) Prototype-based zero-shot (ResNet features from ResNetWrapper compared to stored prototypes).
- Returns rich structured outputs, including:
    - Per-segment predictions from each method
    - Aggregated counts per label
    - Per-model inference times (SAM, ResNet, CLIP)
    - Model names used and metadata
    - Per-label and overall classification metrics when running `evaluate(...)`
- Functions:
    - classify_image(...) : classify an image with candidate textual labels
    - evaluate(...)       : run an evaluation over generated images and return metrics
    - extract_segment_features(...) : helper to get ResNet features for segments
    - load_clip_model(...) : load CLIP model (optional)
    - helper utilities: loading prototypes, computing similarity, etc.

Design notes:
- Each segment is classified independently. The pipeline chooses a final "chosen_label" per segment by preference:
    1) CLIP prediction if enabled and score >= min_confidence
    2) Prototype prediction (if enabled and score >= min_confidence)
    3) Otherwise None (unclassified)
- All returned dicts are JSON-serializable friendly (floats, lists, strings).
"""
from typing import Callable, Dict, Any, List, Optional, Tuple
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image

from ..segmentation.sam_model import SAMWrapper
from ..feature_extractor.resnet_model import ResNetWrapper
from ...utils.metrics import classification_metrics, per_label_metrics
from ...utils.timing import measure_time

# Attempt to import CLIP (optional)
try:
    import clip
    import torch
except Exception:
    clip = None
    torch = None


# -------------------------
# Internal helpers
# -------------------------
def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


# -------------------------
# ZeroShotManager
# -------------------------
class ZeroShotManager:
    """
    Manager for zero-shot classification at the segment level.

    Args:
        sam (True | False): True creates the sam_wrapper object internally.
        resnet (ResNetWrapper | None): ResNet wrapper instance; created if None.
        clip_model_name (str | None): CLIP model name (e.g., "ViT-B/32"). Used when loading CLIP.
        device (str | None): Torch device if using CLIP (defaults to ResNetWrapper device).
    """

    def __init__(self,
                 sam: bool = True,
                 resnet: Optional[ResNetWrapper] = None,
                 clip_model_name: Optional[str] = None,
                 device: Optional[str] = None):
        if sam:
            self.sam = SAMWrapper()
        else:
            self.sam = None

        self.resnet = resnet or ResNetWrapper()
        self.clip_model_name = clip_model_name or None
        # CLIP model will be lazily loaded when needed
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = device or getattr(self.resnet, "device", None)

    # -------------------------
    # CLIP loader
    # -------------------------
    def load_clip(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Load CLIP model and preprocess function.

        Raises:
            ImportError if clip package unavailable.
        """
        if clip is None:
            raise ImportError("CLIP package not found. Install with: pip install git+https://github.com/openai/CLIP.git")
        model_name = model_name or self.clip_model_name or "ViT-B/32"
        device = device or self._clip_device or "cpu"
        self._clip_device = device
        self._clip_model, self._clip_preprocess = clip.load(model_name, device=device)
        self._clip_model.eval()
        self.clip_model_name = model_name
        return self._clip_model, self._clip_preprocess

    # -------------------------
    # Feature extraction for segments (wraps ResNet)
    # -------------------------
    def extract_segment_features(self, segment_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract ResNet features for a single segment image (by file path).

        Returns:
            feat: numpy 1D vector
            metadata: model_name, device, feature_dim and inference_time (added by decorator)
        """
        # ResNetWrapper.extract_features_from_path returns (feat, metadata)
        feat, meta = self.resnet.extract_features_from_path(segment_path)
        return feat, meta

    # -------------------------
    # Prototype helper
    # -------------------------
    def _load_prototypes(self, store_root: str) -> Dict[str, np.ndarray]:
        """
        Load prototypes using the ResNet wrapper format.
        """
        return self.resnet.load_prototypes(store_root)

    def _prototype_similarity(self, feat: np.ndarray, prototypes: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Compute cosine similarity between feature and prototypes.
        Returns list of (label, score) sorted descending.
        """
        if not prototypes:
            return []
        labels = list(prototypes.keys())
        proto_mat = np.vstack([prototypes[l] for l in labels]).astype(np.float32)
        f = feat.astype(np.float32)
        if np.linalg.norm(f) > 0:
            f = f / np.linalg.norm(f)
        sims = np.dot(proto_mat, f)  # (N,)
        order = np.argsort(-sims)
        return [(labels[i], float(sims[i])) for i in order]

    # -------------------------
    # CLIP zero-shot per segment
    # -------------------------
    def _clip_classify_segment(self, segment_image_path: str, candidate_labels: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Classify a segment (PIL image file) using CLIP zero-shot.

        Returns:
            result: { 'labels': candidate_labels, 'probs': [...], 'predicted': top_label, 'top_prob': float }
            meta: { 'model_name', 'device', 'inference_time' }
        """
        if clip is None:
            raise ImportError("CLIP package not installed. Install via: pip install git+https://github.com/openai/CLIP.git")

        # Ensure model loaded
        if self._clip_model is None:
            self.load_clip(model_name=self.clip_model_name, device=self._clip_device)

        # Preprocess and encode
        img = Image.open(segment_image_path).convert("RGB")
        image_input = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)

        prompts = [f"a photo of a {lab}" for lab in candidate_labels]
        text_tokens = clip.tokenize(prompts).to(self._clip_device)

        t0 = time.time()
        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_input)  # 1 x D
            text_features = self._clip_model.encode_text(text_tokens)    # N x D

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ text_features.T).cpu().numpy().reshape(-1)  # N
            # softmax to convert to probabilities
            exps = np.exp(logits - np.max(logits))
            probs = exps / exps.sum()
            top_idx = int(np.argmax(probs))
            predicted = candidate_labels[top_idx]
            top_prob = float(probs[top_idx])
        t_elapsed = time.time() - t0

        result = {"labels": candidate_labels, "probs": [float(p) for p in probs], "predicted": predicted, "top_prob": top_prob}
        meta = {"model_name": f"CLIP-{self.clip_model_name}", "device": self._clip_device, "inference_time": t_elapsed}
        return result, meta

    # -------------------------
    # Core: classify image (segments) with zero-shot methods
    # -------------------------
    @measure_time
    def classify_image(self,
                       image_path: str,
                       candidate_labels: List[str],
                       use_clip: bool = True,
                       use_prototypes: bool = True,
                       prototypes_store: Optional[str] = None,
                       top_k: int = 1,
                       min_confidence: float = 0.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Segment the image with SAM, then classify each segment with CLIP zero-shot and/or prototypes.

        Args:
            image_path: input image path (str)
            candidate_labels: textual candidate labels for CLIP (list[str])
            use_clip: whether to attempt CLIP-based zero-shot. If True and CLIP not installed -> ImportError.
            use_prototypes: whether to attempt prototype-based zero-shot (requires prototypes stored under prototypes_store)
            prototypes_store: path where prototypes are stored (ResNetWrapper finalize_prototypes layout)
            top_k: number of top prototype labels to return per segment
            min_confidence: minimum score to accept a label from any method

        Returns:
            result: {
                "segments": [
                    {
                        "segment_path": ...,
                        "clip": { ... } or None,
                        "prototype": [ (label, score), ... ] or None,
                        "chosen_label": label_or_None,
                        "chosen_source": "clip" or "prototype" or None,
                        "chosen_score": float
                    }, ...
                ],
                "counts": { label: count, ... }
            }
            metadata: {
                "model_sam": sam_model_name,
                "model_resnet": resnet_model_name,
                "model_clip": clip_model_name or None,
                "inference_time": total_elapsed_by_decorator,
                "times": { "sam": float, "resnet_per_segment_avg": float, "clip_per_segment_avg": float, "prototype_per_segment_avg": float },
                "num_segments": int,
                "avg_confidence_clip": float,
                "avg_confidence_prototype": float
            }
        """
        overall_start = time.time()

        # 1) Run SAM segmentation
        sam_res, sam_meta = self.sam.segment_image(image_path)
        segment_paths = sam_res.get("segments", [])
        sam_model_name = sam_res.get("model_name") or getattr(self.sam, "model_type", "SAM")

        # load prototypes if requested
        prototypes = {}
        if use_prototypes:
            if not prototypes_store:
                raise ValueError("prototypes_store path is required when use_prototypes=True")
            prototypes = self._load_prototypes(prototypes_store)

        # prepare accumulators
        segments_out = []
        counts = defaultdict(int)
        clip_times = []
        proto_times = []
        resnet_times = []
        clip_confidences = []
        proto_confidences = []

        # Optionally load CLIP model (lazy)
        if use_clip:
            if clip is None:
                raise ImportError("CLIP package not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            if self._clip_model is None:
                self.load_clip(model_name=self.clip_model_name, device=self._clip_device)

        # Process each segment
        for seg_path in segment_paths:
            # 2) extract resnet features for this segment
            t0_res = time.time()
            feat, feat_meta = self.resnet.extract_features_from_path(seg_path)
            t_res_elapsed = time.time() - t0_res
            resnet_times.append(t_res_elapsed)

            clip_result = None
            clip_meta = None
            if use_clip:
                try:
                    t0c = time.time()
                    clip_result, clip_meta = self._clip_classify_segment(seg_path, candidate_labels)
                    t_clip = time.time() - t0c
                except Exception as e:
                    clip_result = {"error": str(e)}
                    clip_meta = {"inference_time": 0.0}
                    t_clip = 0.0
                clip_times.append(clip_meta.get("inference_time", t_clip))
                if clip_result and isinstance(clip_result, dict) and "top_prob" in clip_result:
                    clip_confidences.append(clip_result["top_prob"])

            proto_preds = None
            if use_prototypes:
                t0p = time.time()
                proto_preds = self._prototype_similarity(feat, prototypes)  # list sorted desc
                t_proto = time.time() - t0p
                proto_times.append(t_proto)
                if proto_preds:
                    proto_confidences.append(proto_preds[0][1])

            # Decide best label for counting:
            chosen_label = None
            chosen_source = None
            chosen_score = 0.0

            # Preference: CLIP (if enabled) then prototypes
            if use_clip and clip_result and isinstance(clip_result, dict) and "predicted" in clip_result:
                if clip_result["top_prob"] >= min_confidence:
                    chosen_label = clip_result["predicted"]
                    chosen_source = "clip"
                    chosen_score = clip_result["top_prob"]
            if chosen_label is None and use_prototypes and proto_preds:
                if proto_preds[0][1] >= min_confidence:
                    chosen_label = proto_preds[0][0]
                    chosen_source = "prototype"
                    chosen_score = proto_preds[0][1]

            if chosen_label:
                counts[chosen_label] += 1

            segments_out.append({
                "segment_path": seg_path,
                "resnet_feature_meta": feat_meta,
                "clip": clip_result,
                "clip_meta": clip_meta,
                "prototype_preds": proto_preds[:top_k] if proto_preds else None,
                "chosen_label": chosen_label,
                "chosen_source": chosen_source,
                "chosen_score": float(chosen_score)
            })

        # finalize metadata
        total_elapsed = time.time() - overall_start
        metadata = {
            "model_sam": sam_model_name,
            "model_resnet": getattr(self.resnet, "model_name", "resnet"),
            "model_clip": getattr(self, "clip_model_name", None),
            "inference_time": total_elapsed,  # main wrapper time (measure_time decorator will add another copy)
            "times": {
                "sam": float(sam_meta.get("inference_time", 0.0)),
                "resnet_per_segment_avg": float(np.mean(resnet_times)) if resnet_times else 0.0,
                "clip_per_segment_avg": float(np.mean(clip_times)) if clip_times else 0.0,
                "prototype_per_segment_avg": float(np.mean(proto_times)) if proto_times else 0.0
            },
            "num_segments": len(segment_paths),
            "avg_confidence_clip": float(np.mean(clip_confidences)) if clip_confidences else 0.0,
            "avg_confidence_prototype": float(np.mean(proto_confidences)) if proto_confidences else 0.0
        }

        result = {"segments": segments_out, "counts": dict(counts)}
        return result, metadata

    # ----------------------------
    # evaluation (no segmentation)
    # ----------------------------
    def evaluate(
            self,
            generate_images_fn: Callable[..., List[str]],
            labels: List[str], n_per_label_test: int,
            candidate_labels: List[str],
            prototypes_store: Optional[str] = None,
            use_clip: bool = True, use_prototypes: bool = True,
            gen_kwargs: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
        """
        Evaluate zero-shot classifiers on a generated test set.

        Args:
            generate_images_fn: function that produces images; must accept named args matching generator signature (num_images, label, ...)
            labels: ground-truth labels to generate (list)
            n_per_label_test: number of images per label to generate for testing
            candidate_labels: candidate textual labels to give to CLIP (list)
            prototypes_store: path to prototypes (for prototype-based method)
            use_clip: whether to evaluate CLIP-based zero-shot
            use_prototypes: whether to evaluate prototype-based zero-shot
            gen_kwargs: extra kwargs to pass to generator
            max_segments_per_image: optional cap on segments per image

        Returns:
            dict with:
                - per_method metrics (overall + per_label)
                - per_segment_results list
                - timing info (avg inference times)
        """
        gen_kwargs = dict(gen_kwargs or {})
        if use_clip and self._clip_model is None:
            self.load_clip()

        y_true_clip, y_pred_clip, y_true_proto, y_pred_proto = [], [], [], []
        per_sample_results = []

        for label in labels:
            test_images = generate_images_fn(num_images=n_per_label_test, label=label, **gen_kwargs)
            for img_path in test_images:
                feat, _ = self.resnet.extract_features_from_path(img_path)
                clip_pred, proto_pred = None, None

                if use_clip:
                    img = Image.open(img_path).convert("RGB")
                    image_input = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)
                    text_tokens = clip.tokenize([f"a photo of a {lab}" for lab in candidate_labels]).to(self._clip_device)
                    with torch.no_grad():
                        im_feat = self._clip_model.encode_image(image_input)
                        txt_feat = self._clip_model.encode_text(text_tokens)
                        im_feat /= im_feat.norm(dim=-1, keepdim=True)
                        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                        sims = (im_feat @ txt_feat.T).cpu().numpy().flatten()
                        pred_idx = int(np.argmax(sims))
                        clip_pred = candidate_labels[pred_idx]
                        y_true_clip.append(label)
                        y_pred_clip.append(clip_pred)

                if use_prototypes and prototypes_store:
                    protos = self.resnet.load_prototypes(prototypes_store)
                    sims = {lab: float(np.dot(feat, protos[lab]) / (np.linalg.norm(feat) * np.linalg.norm(protos[lab]) + 1e-8))
                            for lab in protos}
                    proto_pred = max(sims, key=sims.get)
                    y_true_proto.append(label)
                    y_pred_proto.append(proto_pred)

                per_sample_results.append({"image": str(img_path), "true_label": label,
                                           "clip_pred": clip_pred, "proto_pred": proto_pred})

        metrics = {}
        if y_true_clip:
            metrics["clip"] = {"overall": classification_metrics(y_true_clip, y_pred_clip, average="macro"),
                               "per_label": per_label_metrics(y_true_clip, y_pred_clip)}
        if y_true_proto:
            metrics["prototype"] = {"overall": classification_metrics(y_true_proto, y_pred_proto, average="macro"),
                                    "per_label": per_label_metrics(y_true_proto, y_pred_proto)}
        return {"metrics": metrics, "per_sample_results": per_sample_results}
