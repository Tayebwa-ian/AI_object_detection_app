"""
few_shot_manager.py

FewShotManager (final version)

Capabilities:
- Build dataset from a generator (the generator signature you provided).
- Extract features using a provided feature_extractor (ResNetWrapper or EfficientNetWrapper).
- Build prototypes (cosine) and save them under store_root/prototypes/<label>/prototype.npy
- Train classifiers:
    - logistic regression (sklearn) via `train_logistic_regression`
    - linear probe (PyTorch) via `train_linear_probe`
- Classify segments or whole images with:
    - prototypes (cosine similarity)
    - logistic regression
    - linear probe
- Full test_pipeline (generate train set, build prototypes/train classifier, evaluate on generated test set)
- classify_image for user images (segment with given segmenter, extract features for segments, classify each segment)
- count_objects_in_image (returns counts for requested labels)
- clear_store (safe deletion)

Design notes:
- This manager does NOT auto-create the feature_extractor: you must pass an instance (ResNetWrapper or EfficientNetWrapper).
- The segmentation model may be passed (SAMWrapper or DeepLabV3Wrapper) and will only be used for user image classification.
- Standard docstrings and debug prints are included.
"""

import os
import time
import uuid
import json
import shutil
import logging
from typing import Callable, Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np
import joblib


# Expected external modules / wrappers (adjust import paths if your structure differs)
from src.pipeline.models.segmentation.sam_model import SAMWrapper
from src.pipeline.models.segmentation.deeplabv3_model import DeepLabV3Wrapper
from src.pipeline.models.feature_extractor.resnet_model import ResNetWrapper
from src.pipeline.models.feature_extractor.efficientnet import EfficientNetWrapper

from src.pipeline.models.classifier.classifiers import *

from src.pipeline.utils.metrics import classification_metrics, per_label_metrics


# -------------------------
# Helpers / I/O utilities
# -------------------------
def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_feature_samples_from_store(store_root: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    Load stored features saved under store_root/samples/<label>/sample_*.npy
    Returns:
        X: (N,D) array, y: list of labels length N, metas: list of dicts
    """
    samples_root = os.path.join(store_root, "samples")
    if not os.path.exists(samples_root):
        return np.zeros((0,)), [], []
    X_list, y_list, metas = [], [], []
    for label in os.listdir(samples_root):
        lab_dir = os.path.join(samples_root, label)
        if not os.path.isdir(lab_dir):
            continue
        for fname in os.listdir(lab_dir):
            if fname.startswith("sample_") and fname.endswith(".npy"):
                path = os.path.join(lab_dir, fname)
                try:
                    arr = np.load(path)
                except Exception:
                    continue
                X_list.append(arr.astype(np.float32))
                y_list.append(label)
                meta_path = path.replace(".npy", ".json").replace("sample_", "meta_")
                metas.append({"feature_path": path, "meta_path": meta_path if os.path.exists(meta_path) else None, "label": label})
    if not X_list:
        return np.zeros((0,)), [], []
    return np.vstack(X_list), y_list, metas


# -------------------------
# FewShotManager class
# -------------------------
class FewShotManager:
    """
    Final FewShotManager.

    Args:
        feature_extractor: an instance implementing extract_features_from_path(path)->(feat, meta),
                           and load_prototypes(store_root)->Dict[label, np.ndarray] and finalize_prototypes(store_root)
                           (e.g., ResNetWrapper or EfficientNetWrapper).
        segmentation_model: segmentation wrapper (SAMWrapper or DeepLabV3Wrapper) for user image classification; optional.
        classifier_type: "prototypes", "logistic", or "linear_probe"
        classifier_params: optional dict passed to training functions (e.g., epochs, lr)
    """

    def __init__(self,
                 feature_extractor: Any,
                 segmentation_model: Optional[Any] = None,
                 classifier_type: str = "prototypes",
                 classifier_params: Optional[Dict[str, Any]] = None):
        if feature_extractor is None:
            raise ValueError("feature_extractor instance required (ResNetWrapper or EfficientNetWrapper).")
        self.feature_extractor = feature_extractor
        self.segmentation_model = segmentation_model
        self.classifier_type = classifier_type.lower()
        self.classifier_params = classifier_params or {}

        # state holders
        self.classifier_obj: Optional[Dict] = None
        self.prototypes: Optional[Dict[str, np.ndarray]] = None

        print(f"[FewShotManager] initialized with feature_extractor={getattr(self.feature_extractor,'model_name',None)}, "
              f"classifier_type={self.classifier_type}, segmentation={type(self.segmentation_model).__name__ if self.segmentation_model else None}")

    # -------------------------
    # Dataset build (synthetic) - extracts features for whole images and saves them
    # -------------------------
    def generate_and_build_dataset(self,
                                   label: str,
                                   num_images: int,
                                   image_generator_fn: Callable[..., List[str]],
                                   store_root: str,
                                   gen_kwargs: Optional[Dict[str, Any]] = None,
                                   verbose: bool = True) -> Dict[str, Any]:
        """
        Use provided generator to create images, extract features (whole image) and save under store_root/samples/<label>.
        """
        gen_kwargs = dict(gen_kwargs or {})
        _ensure_dir(store_root)
        samples_dir = os.path.join(store_root, "samples", label)
        _ensure_dir(samples_dir)

        print(f"[FewShot] Generating {num_images} images for label '{label}'")
        generated_paths = image_generator_fn(num_images=num_images, label=label, **gen_kwargs)

        stored = []
        for p in generated_paths:
            feat, meta = self.feature_extractor.extract_features_from_path(str(p))
            sid = str(uuid.uuid4())
            feat_path = os.path.join(samples_dir, f"sample_{sid}.npy")
            meta_path = os.path.join(samples_dir, f"meta_{sid}.json")
            np.save(feat_path, feat.astype(np.float32))
            _save_json({"source_image": str(p), "feature_meta": meta}, meta_path)
            stored.append({"sample_id": sid, "feature_path": feat_path, "meta_path": meta_path, "label": label})
            if verbose:
                print(f"[FewShot] saved sample {feat_path}")
        return {"stored_samples": stored, "num_generated": len(generated_paths)}

    # -------------------------
    # Build and save prototypes (cosine)
    # -------------------------
    def finalize_prototypes(self, store_root: str) -> Dict[str, Any]:
        """
        Build prototypes from stored samples and save them under store_root/prototypes/<label>/prototype.npy
        """
        X, y, metas = _load_feature_samples_from_store(store_root)
        if X.size == 0:
            print("[FewShot] No samples found to build prototypes.")
            return {}
        prototypes = build_prototypes_from_samples(X, y)
        prot_root = os.path.join(store_root, "prototypes")
        _ensure_dir(prot_root)
        info = {}
        for lab, proto in prototypes.items():
            lab_dir = os.path.join(prot_root, lab)
            _ensure_dir(lab_dir)
            ppath = os.path.join(lab_dir, "prototype.npy")
            np.save(ppath, proto.astype(np.float32))
            info[lab] = {"prototype_path": ppath, "dim": int(proto.shape[0])}
            print(f"[FewShot] saved prototype for '{lab}' -> {ppath}")
        self.prototypes = prototypes
        return info

    # -------------------------
    # Train classifier (logistic or linear_probe)
    # -------------------------
    def train_classifier(self, store_root: str, classifier_store_path: str) -> Dict[str, Any]:
        """
        Train classifier according to self.classifier_type.
        Returns metadata and saved paths.
        """
        X, y, metas = _load_feature_samples_from_store(store_root)
        if X.size == 0:
            raise ValueError("No training samples found in store_root.")

        if self.classifier_type == "prototypes":
            proto_info = self.finalize_prototypes(store_root)
            return {"method": "prototypes", "prototypes": proto_info}

        elif self.classifier_type == "logistic":
            print("[FewShot] training logistic regression...")
            trained = train_logistic_regression(X, y, **self.classifier_params)
            os.makedirs(os.path.dirname(classifier_store_path) or ".", exist_ok=True)
            joblib.dump(trained, classifier_store_path)
            self.classifier_obj = trained
            print(f"[FewShot] logistic saved at {classifier_store_path}")
            return {"method": "logistic", "path": os.path.abspath(classifier_store_path), "meta": trained["meta"]}

        elif self.classifier_type == "linear_probe":
            print("[FewShot] training linear probe...")
            weight_path = self.classifier_params.get("weight_path")
            device = self.classifier_params.get("device", "cpu")
            probe = train_linear_probe(X, y,
                                      epochs=int(self.classifier_params.get("epochs", 20)),
                                      lr=float(self.classifier_params.get("lr", 1e-2)),
                                      batch_size=int(self.classifier_params.get("batch_size", 32)),
                                      weight_path=weight_path,
                                      device=device)
            # store probe to joblib for convenience
            os.makedirs(os.path.dirname(classifier_store_path) or ".", exist_ok=True)
            joblib.dump(probe, classifier_store_path)
            self.classifier_obj = probe
            print(f"[FewShot] linear probe saved at {classifier_store_path}")
            return {"method": "linear_probe", "path": os.path.abspath(classifier_store_path), "meta": probe["meta"]}

        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

    # -------------------------
    # Load classifier / prototypes
    # -------------------------
    def load_classifier(self, classifier_store_path: str) -> Any:
        if not os.path.exists(classifier_store_path):
            raise FileNotFoundError(classifier_store_path)
        obj = joblib.load(classifier_store_path)
        self.classifier_obj = obj
        print(f"[FewShot] loaded classifier/probe from {classifier_store_path}")
        return obj

    def load_prototypes(self, store_root: str) -> Dict[str, np.ndarray]:
        prot_root = os.path.join(store_root, "prototypes")
        proxies: Dict[str, np.ndarray] = {}
        if not os.path.exists(prot_root):
            return proxies
        for lab in os.listdir(prot_root):
            ppth = os.path.join(prot_root, lab, "prototype.npy")
            if os.path.exists(ppth):
                proxies[lab] = np.load(ppth)
        self.prototypes = proxies
        return proxies

    # -------------------------
    # classify_image (user image -> segmentation -> features -> classify segments)
    # -------------------------
    def classify_image(self,
                       image_path: str,
                       store_root: str,
                       classifier_store_path: Optional[str] = None,
                       top_k: int = 1,
                       min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Classify an image with segmentation + chosen classification method.

        Returns:
            dict with segments list, counts, metadata (incl. timings and model names).
        """
        if self.segmentation_model is None:
            raise RuntimeError("segmentation_model not provided to FewShotManager for classify_image.")

        run_id = str(uuid.uuid4())
        tstart = time.time()

        seg_res, seg_meta = self.segmentation_model.segment_image(image_path)
        seg_paths = seg_res.get("segments", [])
        seg_model_name = seg_res.get("model_name") if isinstance(seg_res, dict) else type(self.segmentation_model).__name__

        # load classifier/prototypes if necessary
        if classifier_store_path and os.path.exists(classifier_store_path):
            self.load_classifier(classifier_store_path)

        prototypes = self.load_prototypes(store_root) if os.path.exists(os.path.join(store_root, "prototypes")) else {}

        segments_out = []
        counts = defaultdict(int)
        feature_times, cls_times, proto_times = [], [], []

        for sp in seg_paths:
            tf0 = time.time()
            feat, feat_meta = self.feature_extractor.extract_features_from_path(sp)
            feature_times.append(time.time() - tf0)

            chosen_label = None
            chosen_score = 0.0
            chosen_source = None

            # classifier priority: classifier (logistic/linear_probe) > prototypes
            if self.classifier_type == "logistic" and self.classifier_obj:
                tc0 = time.time()
                lab, prob = predict_with_logistic(self.classifier_obj, feat)[0]
                cls_times.append(time.time() - tc0)
                if prob >= min_confidence:
                    chosen_label, chosen_score, chosen_source = lab, prob, "logistic"

            elif self.classifier_type == "linear_probe" and self.classifier_obj:
                tc0 = time.time()
                try:
                    lab, prob = predict_with_linear_probe(self.classifier_obj, feat)
                    cls_times.append(time.time() - tc0)
                    if prob >= min_confidence:
                        chosen_label, chosen_score, chosen_source = lab, prob, "linear_probe"
                except Exception:
                    cls_times.append(time.time() - tc0)

            # prototypes fallback if no classifier label accepted
            proto_preds = None
            if (chosen_label is None) and prototypes:
                tp0 = time.time()
                proto_preds = classify_with_prototypes(feat, prototypes, top_k=top_k)
                proto_times.append(time.time() - tp0)
                if proto_preds and proto_preds[0][1] >= min_confidence:
                    chosen_label, chosen_score, chosen_source = proto_preds[0][0], proto_preds[0][1], "prototype"

            if chosen_label:
                counts[chosen_label] += 1

            segments_out.append({
                "segment_path": sp,
                "feature_meta": feat_meta,
                "classifier_pred": {"label": chosen_label, "score": float(chosen_score), "source": chosen_source},
                "prototype_preds": proto_preds
            })

        metadata = {
            "run_id": run_id,
            "segmentation_model": seg_model_name,
            "feature_model": getattr(self.feature_extractor, "model_name", None),
            "num_segments": len(seg_paths),
            "elapsed_time": time.time() - tstart,
            "avg_feature_time": float(np.mean(feature_times)) if feature_times else None,
            "avg_classifier_time": float(np.mean(cls_times)) if cls_times else None,
            "avg_prototype_time": float(np.mean(proto_times)) if proto_times else None
        }
        print(f"[FewShot] classify_image done run_id={run_id} segments={len(seg_paths)} elapsed={metadata['elapsed_time']:.3f}s")
        return {"segments": segments_out, "counts": dict(counts), "metadata": metadata}

    # -------------------------
    # test_pipeline (complete training + evaluation on synthetic data)
    # -------------------------
    def test_pipeline(self,
                      generate_images_fn: Callable[..., List[str]],
                      labels: List[str],
                      n_per_label_train: int,
                      n_per_label_test: int,
                      store_root: str,
                      classifier_store_path: str,
                      gen_kwargs: Optional[Dict[str, Any]] = None,
                      use_existing_classifier: bool = False,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Full test pipeline:
         - generate train set (features)
         - build prototypes OR train classifier
         - evaluate on generated test images
        Returns structured summary with metrics and run_id.
        """
        run_id = str(uuid.uuid4())
        start_all = time.time()
        gen_kwargs = dict(gen_kwargs or {})

        print(f"[FewShot] test_pipeline start run_id={run_id}")
        train_summary = {}
        # build training dataset
        self.clear_store(store_root=store_root) # clear data stored and build a fresh dataset
        for lab in labels:
            out = self.generate_and_build_dataset(label=lab, num_images=n_per_label_train,
                                                  image_generator_fn=generate_images_fn,
                                                  store_root=store_root, gen_kwargs=gen_kwargs, verbose=verbose)
            train_summary[lab] = out

        # train or build prototypes
        if use_existing_classifier and os.path.exists(classifier_store_path):
            print("[FewShot] using existing classifier/probe")
            clf_info = {"note": "loaded existing", "path": classifier_store_path}
            self.load_classifier(classifier_store_path)
        else:
            clf_info = self.train_classifier(store_root=store_root, classifier_store_path=classifier_store_path)  # type: ignore

        # evaluate
        y_true, y_pred = [], []
        per_sample = []
        times = []

        for lab in labels:
            test_images = generate_images_fn(num_images=n_per_label_test, label=lab, **gen_kwargs)
            for img in test_images:
                t0 = time.time()
                # if classifier exists and is parametric -> use it; else use prototypes
                pred_label = None
                if self.classifier_type in ("logistic", "linear_probe") and self.classifier_obj:
                    if self.classifier_type == "logistic":
                        labp, score = predict_with_logistic(self.classifier_obj, self.feature_extractor.extract_features_from_path(str(img))[0])[0]
                        pred_label = labp
                    else:
                        try:
                            labp, score = predict_with_linear_probe(self.classifier_obj, self.feature_extractor.extract_features_from_path(str(img))[0])
                            pred_label = labp
                        except Exception:
                            pred_label = None
                else:
                    # prototypes
                    protos = self.load_prototypes(store_root)
                    feat, _ = self.feature_extractor.extract_features_from_path(str(img))
                    proto_preds = classify_with_prototypes(feat, protos, top_k=1)
                    pred_label = proto_preds[0][0] if proto_preds else None

                times.append(time.time() - t0)
                y_true.append(lab)
                y_pred.append(pred_label)
                per_sample.append({"image": str(img), "true": lab, "pred": pred_label})

        metrics = {"overall": classification_metrics(y_true, y_pred, average="macro"),
                   "per_label": per_label_metrics(y_true, y_pred)}
        total_elapsed = time.time() - start_all
        summary = {
            "run_id": run_id,
            "train_summary": train_summary,
            "classifier_info": clf_info,
            "metrics": metrics,
            "num_test_samples": len(y_true),
            "avg_inference_time": float(np.mean(times)) if times else None,
            "total_elapsed_time": total_elapsed
        }
        print(f"[FewShot] test_pipeline done run_id={run_id} elapsed={total_elapsed:.3f}s")
        return {"summary": summary, "per_sample_results": per_sample}

    # -------------------------
    # count_objects_in_image
    # -------------------------
    def count_objects_in_image(self, image_path: str, store_root: str, classifier_store_path: Optional[str] = None,
                               target_labels: Optional[List[str]] = None, min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Count objects in user image. Returns counts limited to target_labels if provided.
        """
        out = self.classify_image(image_path=image_path,
                                  store_root=store_root,
                                  classifier_store_path=classifier_store_path,
                                  top_k=1,
                                  min_confidence=min_confidence)
        counts = out["counts"]
        if target_labels:
            filtered = {lab: counts.get(lab, 0) for lab in target_labels}
        else:
            filtered = counts
        return {"counts": filtered, "segments": out["segments"], "metadata": out["metadata"]}

    
    def clear_store(self, store_root: str) -> None:
        """
        Clear only the 'prototypes', 'classifiers', and 'data' folders inside store_root.
        This is useful when rebuilding a new training dataset while preserving stored samples.

        Args:
            store_root (str): Root directory where all few-shot data is stored.

        Raises:
            ValueError: If store_root is invalid or a critical system path.
        """
        # Ensure we are working with a string
        store_root = str(store_root)

        if not store_root:
            raise ValueError("store_root cannot be empty")

        # Safety: prevent accidental system deletion
        critical_paths = ["/", "/home", "/etc", "C:\\", "C:\\Windows"]
        if store_root.rstrip(os.sep) in critical_paths:
            raise ValueError(f"Refusing to clear critical path: {store_root}")

        targets = ["samples"]
        for folder in targets:
            path = os.path.join(store_root, folder)
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    logging.info(f"[FewShot] Cleared {path}")
                except OSError as e:
                    logging.error(f"Failed to clear {path}: {e}")
                    raise
            else:
                logging.debug(f"[FewShot] Skipped missing folder: {path}")


