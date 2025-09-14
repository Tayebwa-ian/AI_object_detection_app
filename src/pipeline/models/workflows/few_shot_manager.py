"""
few_shot_manager.py

Updated FewShotManager that integrates with an image generator having signature:

def generate_images(
    num_images: int = DEFAULT_NUM_IMAGES,
    label: str = DEFAULT_LABEL,
    model: str = DEFAULT_MODEL,
    sampling_steps: int = DEFAULT_SAMPLING_STEPS,
    sample_method: str = DEFAULT_SAMPLE_METHOD,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    guidance: float = DEFAULT_GUIDANCE,
    strength: float = DEFAULT_STRENGTH,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    example_image_path: Optional[str] = DEFAULT_EXAMPLE_IMAGE_PATH,
) -> List[Path]:

This manager:
 - builds dataset using the provided generator
 - extracts segments with SAM and features with ResNetWrapper
 - stores per-sample features for later prototype/few-shot use
 - trains/loads classifiers (sklearn-style)
 - classifies segments using classifier and/or prototype cosine similarity
 - provides testing and counting utilities

Important:
 - If a classifier_path is provided to classification/testing functions,
   the manager will load the classifier from disk (so you can provide a pre-trained classifier).
"""
from typing import Callable, Dict, Any, List, Optional, Tuple
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ..feature_extractor.resnet_model import ResNetWrapper
from ..segmentation.sam_model import SAMWrapper
from ...utils.metrics import classification_metrics, per_label_metrics

import logging
import shutil

# ----------------------------
# Internal helpers
# ----------------------------
def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def _load_feature_samples_from_store(store_root: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    Load all feature samples saved by ResNetWrapper under store_root/samples/<label>/sample_*.npy.

    Returns:
        X: np.ndarray shape (N, D)
        y: list of labels length N
        metas: list of metadata dicts for each sample
    """
    samples_root = os.path.join(store_root, "samples")
    if not os.path.exists(samples_root):
        return np.zeros((0,)), [], []
    X_list = []
    y_list = []
    metas = []
    for label in os.listdir(samples_root):
        lab_dir = os.path.join(samples_root, label)
        if not os.path.isdir(lab_dir):
            continue
        for f in os.listdir(lab_dir):
            if f.startswith("sample_") and f.endswith(".npy"):
                p = os.path.join(lab_dir, f)
                try:
                    arr = np.load(p)
                except Exception:
                    continue
                X_list.append(arr.astype(np.float32))
                y_list.append(label)
                meta_path = p.replace(".npy", ".json").replace("sample_", "meta_")
                metas.append({"sample_path": p, "meta_path": meta_path if os.path.exists(meta_path) else None, "label": label})
    if not X_list:
        return np.zeros((0,)), [], []
    X = np.vstack(X_list)
    return X, y_list, metas

# ----------------------------
# FewShotManager
# ----------------------------
class FewShotManager:
    """
    Manager for few-shot dataset generation, training, classification, testing and counting.

    Parameters
    ----------
    resnet_wrapper: ResNetWrapper instance (if None, created internally)
    sam_wrapper: SAMWrapper instance (if None, created internally)
    classifier_class: sklearn-like estimator class (default LogisticRegression)
    """
    def __init__(self,
                 resnet_wrapper: Optional[ResNetWrapper] = None,
                 sam_wrapper: Optional[SAMWrapper] = None,
                 classifier_class = LogisticRegression):
        self.resnet = resnet_wrapper or ResNetWrapper()
        self.sam = sam_wrapper or SAMWrapper()
        self.classifier_class = classifier_class

    # ----------------------------
    # Dataset generation (no segmentation)
    # ----------------------------
    def generate_and_build_dataset(self,
                                   label: str,
                                   num_images: int,
                                   image_generator_fn: Callable[..., List[Path]],
                                   store_root: str,
                                   gen_kwargs: Optional[Dict[str, Any]] = None,
                                   verbose: bool = True) -> Dict[str, Any]:
        """
        Generate synthetic images and build dataset by extracting features (no segmentation).
        """
        t0 = time.time()
        _ensure_dir(store_root)
        gen_kwargs = dict(gen_kwargs or {})

        generated_paths = image_generator_fn(num_images=num_images, label=label, **gen_kwargs)

        stored_samples = []
        for img_path in generated_paths:
            if verbose:
                print(f"[dataset] processing generated image: {img_path}")
            sample_meta = self.resnet.add_prototype(label=label, image_path=str(img_path), store_root=store_root)
            stored_samples.append(sample_meta)

        elapsed = time.time() - t0
        summary = {"generated_images": len(generated_paths), "stored_samples": len(stored_samples),
                   "store_root": os.path.abspath(store_root), "elapsed_time": elapsed}
        return {"summary": summary, "stored_samples": stored_samples}

    # ----------------------------
    # Train & Save classifier
    # ----------------------------
    def train_classifier(self,
                         store_root: str,
                         classifier_path: str,
                         classifier_params: Optional[Dict[str, Any]] = None,
                         overwrite: bool = True) -> Dict[str, Any]:
        """
        Train a classifier (sklearn-style) on stored features under store_root and save it.

        Returns:
            dict with metadata about the trained classifier.
        """
        X, y, _ = _load_feature_samples_from_store(store_root)
        if X.size == 0:
            raise ValueError("No samples found under store_root. Generate dataset first.")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        params = classifier_params or {}
        clf = self.classifier_class(**params)
        clf.fit(X, y_enc)

        # save classifier + label encoder + model_name
        os.makedirs(os.path.dirname(classifier_path) or ".", exist_ok=True)
        if os.path.exists(classifier_path) and not overwrite:
            raise FileExistsError(f"{classifier_path} exists and overwrite=False")

        joblib.dump({"classifier": clf, "label_encoder": le, "model_name": getattr(self.resnet, "model_name", "resnet")}, classifier_path)

        return {"num_samples": int(X.shape[0]), "classes": list(le.classes_), "classifier_path": os.path.abspath(classifier_path)}

    # ----------------------------
    # Load classifier
    # ----------------------------
    def load_classifier(self, classifier_path: str) -> Dict[str, Any]:
        """
        Load a saved classifier file previously created by train_classifier.
        Returns a dict with keys: classifier, label_encoder, model_name
        """
        obj = joblib.load(classifier_path)
        return obj

    # ----------------------------
    # Internal classify helpers
    # ----------------------------
    def _classify_feature_with_classifier(self, feature: np.ndarray, classifier_obj: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Predict label and confidence for a single feature with the provided classifier object (loaded).
        Returns (pred_label or None, score).
        """
        clf = classifier_obj["classifier"]
        le = classifier_obj["label_encoder"]
        if not hasattr(clf, "predict_proba"):
            pred_idx = int(clf.predict(feature.reshape(1, -1))[0])
            pred_label = le.inverse_transform([pred_idx])[0]
            return pred_label, 1.0
        probs = clf.predict_proba(feature.reshape(1, -1))[0]
        top_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([top_idx])[0]
        return pred_label, float(probs[top_idx])

    def _classify_with_prototypes_for_feature(self, feature: np.ndarray, store_root: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Return top_k (label, similarity) pairs between feature and prototypes stored in store_root.
        """
        protos = self.resnet.load_prototypes(store_root)
        if not protos:
            return []
        labels = list(protos.keys())
        proto_mat = np.vstack([protos[l] for l in labels]).astype(np.float32)
        f = feature.astype(np.float32)
        if np.linalg.norm(f) > 0:
            f = f / np.linalg.norm(f)
        sims = np.dot(proto_mat, f)
        order = np.argsort(-sims)
        return [(labels[i], float(sims[i])) for i in order[:top_k]]

    # ----------------------------
    # classify_image: uses pre-trained classifier if provided (loads it)
    # ----------------------------
    def classify_image(self,
                       image_path: str,
                       store_root: str,
                       classifier_path: Optional[str] = None,
                       top_k: int = 1,
                       min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Segment input image with SAM, extract ResNet features for segments, classify with:
         - loaded classifier (if classifier_path provided)
         - prototypes (if prototypes exist under store_root)

        Returns:
            dict with segments list (each entry contains classifier and prototype predictions),
            counts aggregated by chosen label, and metadata with timing info.
        """
        t0 = time.time()

        # Run segmentation (sam wrapper saves segments to disk)
        seg_result, seg_meta = self.sam.segment_image(image_path)
        seg_paths = seg_result.get("segments", [])

        classifier_obj = None
        if classifier_path:
            classifier_obj = self.load_classifier(classifier_path)

        segments_out = []
        counts = defaultdict(int)
        times = {"per_segment_feat": [], "per_segment_classifier": [], "per_segment_prototype": []}

        for seg_path in seg_paths:
            # Extract features for segment
            t_feat0 = time.time()
            feat, feat_meta = self.resnet.extract_features_from_path(seg_path)
            t_feat = time.time() - t_feat0
            times["per_segment_feat"].append(t_feat)

            # classifier prediction (if available)
            classifier_pred = None
            tcls0 = time.time()
            if classifier_obj:
                try:
                    lab, score = self._classify_feature_with_classifier(feat, classifier_obj)
                    classifier_pred = {"label": lab, "score": float(score)}
                except Exception as e:
                    classifier_pred = {"label": None, "score": 0.0, "error": str(e)}
            tcls = time.time() - tcls0
            times["per_segment_classifier"].append(tcls)

            # prototype predictions
            tproto0 = time.time()
            prot_preds = self._classify_with_prototypes_for_feature(feat, store_root, top_k=top_k)
            tproto = time.time() - tproto0
            times["per_segment_prototype"].append(tproto)

            # choose label: classifier if meets confidence else prototype top-1 if meets confidence
            chosen_label = None
            chosen_score = 0.0
            if classifier_pred and classifier_pred.get("label") is not None and classifier_pred.get("score", 0.0) >= min_confidence:
                chosen_label = classifier_pred["label"]
                chosen_score = classifier_pred["score"]
            elif prot_preds and prot_preds[0][1] >= min_confidence:
                chosen_label = prot_preds[0][0]
                chosen_score = prot_preds[0][1]

            if chosen_label:
                counts[chosen_label] += 1

            segments_out.append({
                "segment_path": seg_path,
                "classifier_pred": classifier_pred,
                "prototype_preds": prot_preds,
                "chosen_label": chosen_label,
                "chosen_score": chosen_score,
                "segment_meta": feat_meta
            })

        total_time = time.time() - t0
        metadata = {
            "model_sam": seg_result.get("model_name"),
            "num_segments": len(seg_paths),
            "elapsed_time": total_time,
            "avg_time_per_segment_feat": float(np.mean(times["per_segment_feat"])) if times["per_segment_feat"] else None,
            "avg_time_per_segment_classifier": float(np.mean(times["per_segment_classifier"])) if times["per_segment_classifier"] else None,
            "avg_time_per_segment_prototype": float(np.mean(times["per_segment_prototype"])) if times["per_segment_prototype"] else None
        }

        return {"segments": segments_out, "counts": dict(counts), "metadata": metadata}

    # ----------------------------
    # Testing pipeline (no segmentation)
    # ----------------------------
    def test_pipeline(self,
                      generate_images_fn: Callable[..., List[Path]],
                      labels: List[str],
                      n_per_label_train: int,
                      n_per_label_test: int,
                      store_root: str,
                      classifier_path: str,
                      classifier_params: Optional[Dict[str, Any]] = None,
                      gen_kwargs: Optional[Dict[str, Any]] = None,
                      use_existing_classifier: bool = False,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Training/testing pipeline:
        - Build dataset with generated images (no segmentation).
        - Train or load classifier.
        - Evaluate on new generated images.
        """
        overall_start = time.time()
        gen_kwargs = dict(gen_kwargs or {})

        # Training dataset
        train_summary = {}
        for label in labels:
            out = self.generate_and_build_dataset(label=label, num_images=n_per_label_train,
                                                  image_generator_fn=generate_images_fn,
                                                  store_root=store_root, gen_kwargs=gen_kwargs, verbose=verbose)
            train_summary[label] = out["summary"]

        # Prototypes
        proto_info = self.resnet.finalize_prototypes(store_root)

        # Classifier
        if use_existing_classifier and os.path.exists(classifier_path):
            if verbose:
                print("[test] Using existing classifier")
            clf_info = {"classifier_path": os.path.abspath(classifier_path), "note": "loaded existing"}
        else:
            clf_info = self.train_classifier(store_root=store_root, classifier_path=classifier_path,
                                             classifier_params=classifier_params)

        # Evaluation
        y_true, y_pred = [], []
        per_sample_results, times = [], []
        classifier_obj = self.load_classifier(classifier_path)

        for label in labels:
            test_images = generate_images_fn(num_images=n_per_label_test, label=label, **gen_kwargs)
            for img_path in test_images:
                feat, feat_meta = self.resnet.extract_features_from_path(str(img_path))
                t0 = time.time()
                pred_idx = classifier_obj["classifier"].predict(feat.reshape(1, -1))[0]
                pred_label = classifier_obj["label_encoder"].inverse_transform([pred_idx])[0]
                times.append(time.time() - t0)

                y_true.append(label)
                y_pred.append(pred_label)
                per_sample_results.append({"test_image": str(img_path), "true_label": label,
                                           "pred_label": pred_label, "feat_meta": feat_meta})

        metrics = {"overall": classification_metrics(y_true, y_pred, average="macro"),
                   "per_label": per_label_metrics(y_true, y_pred)}
        total_elapsed = time.time() - overall_start
        summary = {"train_summary": train_summary, "prototypes": proto_info, "classifier_info": clf_info,
                   "metrics": metrics, "num_test_samples": len(y_true),
                   "avg_inference_time": float(np.mean(times)) if times else None,
                   "total_elapsed_time": total_elapsed}
        return {"summary": summary, "per_sample_results": per_sample_results}

    # ----------------------------
    # Count objects in an image (uses classifier if provided)
    # ----------------------------
    def count_objects_in_image(self,
                               image_path: str,
                               store_root: str,
                               classifier_path: Optional[str] = None,
                               target_labels: Optional[List[str]] = None,
                               min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Count objects in image for provided target_labels (if None, count all predicted labels).
        Uses classifier (if classifier_path provided) else prototypes.

        Returns:
            dict containing counts mapping and per-segment details.
        """
        res = self.classify_image(image_path=image_path,
                                  store_root=store_root,
                                  classifier_path=classifier_path,
                                  top_k=1,
                                  min_confidence=min_confidence)
        counts = res["counts"]
        if target_labels is not None:
            filtered = {lab: counts.get(lab, 0) for lab in target_labels}
        else:
            filtered = counts
        return {"counts": filtered, "segments": res["segments"], "metadata": res["metadata"]}

    # ----------------------------
    # Utility: delete store
    # ----------------------------
    def clear_store(self, store_root: str) -> None:
        """
        Delete the final directory and all its contents. Parent directories are preserved.
        Use with caution as this operation is irreversible.
        
        Args:
            store_root (str): Path to the directory to be deleted.
        
        Raises:
            ValueError: If store_root is a critical system path.
            OSError: If deletion fails due to permissions or other issues.
        """
        # Safety check to prevent deleting critical paths
        critical_paths = ["/", "/home", "/etc", "C:\\", "C:\\Windows"]
        if store_root.rstrip(os.sep) in critical_paths or not store_root:
            raise ValueError(f"Cannot delete critical or empty path: {store_root}")

        # Check if the path exists and is a directory
        if not os.path.exists(store_root):
            logging.warning(f"Directory does not exist: {store_root}")
            return
        if not os.path.isdir(store_root):
            logging.warning(f"Path is not a directory: {store_root}")
            return

        try:
            # Log the deletion for auditing
            logging.info(f"Deleting directory and contents: {store_root}")
            shutil.rmtree(store_root)
            logging.info(f"Successfully deleted: {store_root}")
        except OSError as e:
            logging.error(f"Error deleting {store_root}: {e}")
            raise
