"""
zero_shot_manager.py (final)

ZeroShotManager - modular and explicit.

Design principles:
- Manager does NOT auto-create ResNet/CLIP; you must pass instances (or None) to the constructor.
- segmenter: segmentation wrapper (SAMWrapper/DeepLabV3Wrapper) or None
- feature_extractor: ResNet/EfficientNet wrapper or None (used for prototypes)
- clip_model & clip_preprocess: objects returned by clip.load (or None) for CLIP-based zero-shot

Capabilities:
- classify_image: segment => per-segment classification with CLIP and/or prototypes
- evaluate: synthetic evaluation on generated images (no segmentation) using CLIP and/or prototypes
- returns run_id, metrics, per-sample and per-segment outputs, times and model names
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

import numpy as np
from PIL import Image

# Optional CLIP imports will only be required if clip_model passed
try:
    import clip  # type: ignore
    import torch  # type: ignore
except Exception:
    clip = None
    torch = None

from src.pipeline.utils.metrics import classification_metrics, per_label_metrics  # type: ignore


class ZeroShotManager:
    """
    ZeroShotManager constructor arguments:
        segmenter: segmentation wrapper instance (SAMWrapper / DeepLabV3Wrapper) or None
        feature_extractor: ResNetWrapper/EfficientNetWrapper or None
        clip_model: CLIP model instance (from clip.load) or None
        clip_preprocess: CLIP preprocess function or None
        device: device string for CLIP inference ("cpu" or "cuda")
    """

    def __init__(self,
                 segmenter: Optional[Any] = None,
                 feature_extractor: Optional[Any] = None,
                 clip_model: Optional[Any] = None,
                 clip_preprocess: Optional[Any] = None,
                 device: str = "cpu"):
        self.segmenter = segmenter
        self.feature_extractor = feature_extractor
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.clip_model_name = getattr(clip_model, "__class__", None).__name__ if clip_model else None
        print(f"[ZeroShotManager] initialized segmentation={type(segmenter).__name__ if segmenter else None}, "
              f"feature_extractor={getattr(feature_extractor,'model_name',None)}, clip_provided={bool(clip_model)}")

    # -------------------------
    # Internal: prototype similarity
    # -------------------------
    def _prototype_similarity(self, feat: np.ndarray, prototypes: Dict[str, np.ndarray]) -> List[tuple]:
        if not prototypes:
            return []
        labels = list(prototypes.keys())
        proto_mat = np.vstack([prototypes[l] for l in labels]).astype(np.float32)
        f = feat.astype(np.float32)
        fn = np.linalg.norm(f)
        if fn > 0:
            f = f / fn
        sims = proto_mat.dot(f)
        order = np.argsort(-sims)
        return [(labels[i], float(sims[i])) for i in order]

    # -------------------------
    # classify a single segment with CLIP (if clip_model provided)
    # -------------------------
    def _clip_classify_segment(self, segment_path: str, candidate_labels: List[str]) -> Dict:
        if self.clip_model is None or self.clip_preprocess is None:
            raise RuntimeError("clip_model and clip_preprocess must be provided for CLIP classification")
        img = Image.open(segment_path).convert("RGB")
        image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        prompts = [f"a photo of a {lab}" for lab in candidate_labels]
        text_tokens = clip.tokenize(prompts).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            im_feat = self.clip_model.encode_image(image_input)
            txt_feat = self.clip_model.encode_text(text_tokens)
            im_feat /= im_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            sims = (im_feat @ txt_feat.T).cpu().numpy().reshape(-1)
            exps = np.exp(sims - np.max(sims))
            probs = exps / exps.sum()
            top_idx = int(np.argmax(probs))
            predicted = candidate_labels[top_idx]
            top_prob = float(probs[top_idx])
        elapsed = time.time() - t0
        return {"labels": candidate_labels, "probs": probs.tolist(), "predicted": predicted, "top_prob": top_prob, "inference_time": elapsed}

    # -------------------------
    # classify_image (user image -> segmentation -> per-segment classify)
    # -------------------------
    def classify_image(self,
                       image_path: str,
                       candidate_labels: List[str],
                       use_clip: bool = True,
                       use_prototypes: bool = True,
                       prototypes_store: Optional[str] = None,
                       top_k: int = 1,
                       min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Segment image (if segmenter provided), then classify segments using CLIP and/or prototypes.

        Returns:
            dict with 'run_id', 'segments' list, 'counts', and 'metadata'.
        """
        run_id = str(uuid.uuid4())
        t_all = time.time()

        # Segment
        if self.segmenter:
            seg_res, seg_meta = self.segmenter.segment_image(image_path)
            segment_paths = seg_res.get("segments", [])
            seg_model_name = seg_res.get("model_name", getattr(self.segmenter, "model_type", type(self.segmenter).__name__))
            seg_time = float(seg_meta.get("inference_time", 0.0) if isinstance(seg_meta, dict) else 0.0)
        else:
            # no segmentation: treat the whole image as one "segment"
            segment_paths = [image_path]
            seg_model_name = None
            seg_time = 0.0

        # load prototypes if requested
        prototypes: Dict[str, np.ndarray] = {}
        if use_prototypes and prototypes_store:
            prot_root = os.path.join(prototypes_store, "prototypes")
            if os.path.exists(prot_root):
                for lab in os.listdir(prot_root):
                    ppth = os.path.join(prot_root, lab, "prototype.npy")
                    if os.path.exists(ppth):
                        prototypes[lab] = np.load(ppth)

        segments_out = []
        counts = defaultdict(int)
        resnet_times, clip_times, proto_times = [], [], []
        clip_confidences, proto_confidences = [], []

        for sp in segment_paths:
            # feature extraction if available
            feat, feat_meta = (None, {})
            if self.feature_extractor:
                t0f = time.time()
                feat, feat_meta = self.feature_extractor.extract_features_from_path(sp)
                resnet_times.append(time.time() - t0f)

            clip_result = None
            if use_clip and self.clip_model:
                try:
                    t0c = time.time()
                    clip_result = self._clip_classify_segment(sp, candidate_labels)
                    clip_times.append(time.time() - t0c)
                    if clip_result and "top_prob" in clip_result:
                        clip_confidences.append(float(clip_result["top_prob"]))
                except Exception as e:
                    clip_result = {"error": str(e)}
                    clip_times.append(0.0)

            proto_preds = None
            if use_prototypes and prototypes and feat is not None:
                t0p = time.time()
                proto_preds = self._prototype_similarity(feat, prototypes)
                proto_times.append(time.time() - t0p)
                if proto_preds:
                    proto_confidences.append(proto_preds[0][1])

            # choose label (preference CLIP then prototype)
            chosen_label, chosen_source, chosen_score = None, None, 0.0
            if use_clip and clip_result and "top_prob" in clip_result and clip_result["top_prob"] >= min_confidence:
                chosen_label, chosen_source, chosen_score = clip_result["predicted"], "clip", float(clip_result["top_prob"])
            elif use_prototypes and proto_preds and proto_preds[0][1] >= min_confidence:
                chosen_label, chosen_source, chosen_score = proto_preds[0][0], "prototype", proto_preds[0][1]

            if chosen_label:
                counts[chosen_label] += 1

            segments_out.append({
                "segment_path": sp,
                "feature_meta": feat_meta,
                "clip": clip_result,
                "prototype_preds": proto_preds[:top_k] if proto_preds else None,
                "chosen_label": chosen_label,
                "chosen_source": chosen_source,
                "chosen_score": float(chosen_score)
            })

        metadata = {
            "run_id": run_id,
            "segmenter": seg_model_name,
            "feature_extractor": getattr(self.feature_extractor, "model_name", None),
            "clip_model": getattr(self.clip_model, "__class__", None).__name__ if self.clip_model else None,
            "inference_time": time.time() - t_all,
            "times": {
                "segmentation": seg_time,
                "resnet_per_segment_avg": float(np.mean(resnet_times)) if resnet_times else None,
                "clip_per_segment_avg": float(np.mean(clip_times)) if clip_times else None,
                "prototype_per_segment_avg": float(np.mean(proto_times)) if proto_times else None
            },
            "num_segments": len(segment_paths),
            "avg_confidence_clip": float(np.mean(clip_confidences)) if clip_confidences else None,
            "avg_confidence_prototype": float(np.mean(proto_confidences)) if proto_confidences else None
        }

        return {"summary": {"run_id": run_id, "counts": dict(counts)}, "segments": segments_out, "metadata": metadata}

    # -------------------------
    # evaluate (synthetic images; no segmentation)
    # -------------------------
    def evaluate(self,
                 generate_images_fn: Callable[..., List[str]],
                 labels: List[str],
                 n_per_label_test: int,
                 candidate_labels: List[str],
                 prototypes: Optional[Dict[str, np.ndarray]] = None,
                 use_clip: bool = True,
                 use_prototypes: bool = True,
                 gen_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate zero-shot on generated images (no segmentation).
        Returns metrics (per method) and per-sample results.
        """
        run_id = str(uuid.uuid4())
        gen_kwargs = dict(gen_kwargs or {})

        y_true_clip, y_pred_clip, y_true_proto, y_pred_proto = [], [], [], []
        per_sample_results = []

        for lab in labels:
            imgs = generate_images_fn(num_images=n_per_label_test, label=lab, **gen_kwargs)
            for img in imgs:
                # feature for prototypes
                feat, feat_meta = (None, {})
                if self.feature_extractor:
                    feat, feat_meta = self.feature_extractor.extract_features_from_path(str(img))

                preds = []

                # CLIP
                if use_clip and self.clip_model and self.clip_preprocess:
                    img_pil = Image.open(img).convert("RGB")
                    image_input = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
                    prompts = [f"a photo of a {c}" for c in candidate_labels]
                    text_tokens = clip.tokenize(prompts).to(self.device)
                    with torch.no_grad():
                        im_feat = self.clip_model.encode_image(image_input)
                        txt_feat = self.clip_model.encode_text(text_tokens)
                        im_feat /= im_feat.norm(dim=-1, keepdim=True)
                        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                        sims = (im_feat @ txt_feat.T).cpu().numpy().flatten()
                        pred_clip = candidate_labels[int(np.argmax(sims))]
                        preds.append(("clip", pred_clip, float(np.max(sims))))
                        y_true_clip.append(lab)
                        y_pred_clip.append(pred_clip)

                # prototypes
                if use_prototypes and prototypes and feat is not None:
                    proto_preds = self._prototype_similarity(feat, prototypes)
                    if proto_preds:
                        preds.append(("prototype", proto_preds[0][0], proto_preds[0][1]))
                        y_true_proto.append(lab)
                        y_pred_proto.append(proto_preds[0][0])

                final = preds[0][1] if preds else None
                per_sample_results.append({"image": str(img), "true_label": lab, "preds": preds, "final": final})

        metrics = {}
        if y_true_clip:
            metrics["clip"] = {"overall": classification_metrics(y_true_clip, y_pred_clip, average="macro"),
                               "per_label": per_label_metrics(y_true_clip, y_pred_clip)}
        if y_true_proto:
            metrics["prototype"] = {"overall": classification_metrics(y_true_proto, y_pred_proto, average="macro"),
                                    "per_label": per_label_metrics(y_true_proto, y_pred_proto)}

        return {"run_id": run_id, "metrics": metrics, "per_sample_results": per_sample_results}
