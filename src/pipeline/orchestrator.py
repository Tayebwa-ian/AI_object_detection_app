"""
orchestrator.py (final)

Unified orchestrator for few-shot and zero-shot workflows.

Allows you to select:
- segmentation model: "sam", "deeplab", or None
- feature extractor: "resnet", "efficientnet" or None
- few-shot classifier: "prototypes", "logistic", "linear_probe"
- zero-shot options: use CLIP (pass clip_model & preprocess) and/or prototypes

This file instantiates the requested models and passes instances into managers.
Each run returns a structured dict containing run_id, results, and metadata.
"""

import os
import uuid
import numpy as np
from typing import Optional, List, Callable, Dict, Any

# Import wrappers
from src.pipeline.models.segmentation.sam_model import SAMWrapper
from src.pipeline.models.segmentation.deeplabv3_model import DeepLabV3Wrapper
from src.pipeline.models.feature_extractor.resnet_model import ResNetWrapper
from src.pipeline.models.feature_extractor.efficientnet import EfficientNetWrapper

from src.pipeline.models.workflows.few_shot_manager import FewShotManager
from src.pipeline.models.workflows.zero_shot_manager import ZeroShotManager
from src.pipeline.config import DEFAULT_CLASSIFIER_PATH, DEFAULT_OUTPUT_ROOT, DEVICE, SAM_CHECKPOINT, LINEAR_SVC_CLASSIFIER_PATH, LOGISTIC_CLASSIFIER_PATH

# The image generator
from src.synthimage.generator import generate_images


# -------------------------
# Factories
# -------------------------
def _instantiate_segmentation(name: Optional[str], sam_checkpoint: Optional[str] = None):
    if name is None:
        return None
    name = name.lower()
    if name == "sam":
        print("[orchestrator] instantiate SAMWrapper")
        # assume SAMWrapper can accept checkpoint path (if implemented)
        if sam_checkpoint:
            return SAMWrapper(checkpoint=sam_checkpoint)
        return SAMWrapper()
    elif name == "deeplab":
        print("[orchestrator] instantiate DeepLabV3Wrapper")
        return DeepLabV3Wrapper()
    else:
        raise ValueError(f"Unknown segmentation model '{name}'")


def _instantiate_feature_extractor(name: Optional[str], device: str = "cpu"):
    if name is None:
        return None
    name = name.lower()
    if name == "resnet":
        print("[orchestrator] instantiate ResNetWrapper")
        return ResNetWrapper(device=device)
    elif name == "efficientnet":
        print("[orchestrator] instantiate EfficientNetWrapper")
        return EfficientNetWrapper(device=device)
    else:
        raise ValueError(f"Unknown feature extractor '{name}'")


# -------------------------
# Orchestrate wrapper
# -------------------------
def orchestrate(
    mode: str,
    # selection args
    segmentation_model_name: Optional[str] = None,
    feature_extractor_name: Optional[str] = "resnet",
    few_shot_classifier_type: str = "logistic",
    use_existing_classifier: bool = False,
    # SAM checkpoint (optional)
    sam_checkpoint: Optional[str] = None,
    # CLIP (for zero-shot) - pass clip.load() outputs if you want to use clip:
    clip_model: Optional[Any] = None,
    clip_preprocess: Optional[Any] = None,
    clip_device: str = DEVICE,
    # common runner args
    generate_images_fn: Optional[Callable[..., List[str]]] = generate_images,
    labels: Optional[List[str]] = None,
    n_per_label_train: Optional[int] = None,
    n_per_label_test: int = 10,
    candidate_labels: Optional[List[str]] = None,
    store_root: str = DEFAULT_OUTPUT_ROOT,
    classifier_store_path: str = DEFAULT_CLASSIFIER_PATH,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    classifier_params: Optional[Dict[str, Any]] = None,
    use_clip: bool = True,
    use_prototypes: bool = True,
    image_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Top-level orchestrator.

    Modes:
        - "few_shot" : synthetic train+test (uses generate_images_fn)
        - "zero_shot": synthetic evaluation (uses generate_images_fn)
        - "user_few_shot": classify one user image (segmentation applied)
        - "user_zero_shot": classify one user image (segmentation applied)
    """

    run_id = str(uuid.uuid4())
    print(f"[orchestrator] run_id={run_id} mode={mode} seg={segmentation_model_name} feat={feature_extractor_name} clf={few_shot_classifier_type}")

    # instantiate requested components
    seg_inst = _instantiate_segmentation(segmentation_model_name, sam_checkpoint)
    feat_inst = _instantiate_feature_extractor(feature_extractor_name, device=clip_device)

    # MODE: few_shot synthetic train & test (no segmentation required)
    if mode == "few_shot":
        if generate_images_fn is None or not labels or n_per_label_train is None:
            raise ValueError("few_shot mode requires generate_images_fn, labels and n_per_label_train")

        fsm = FewShotManager(feature_extractor=feat_inst, segmentation_model=seg_inst,
                             classifier_type=few_shot_classifier_type, classifier_params=classifier_params or {})
        result = fsm.test_pipeline(generate_images_fn=generate_images_fn,
                                   labels=labels,
                                   n_per_label_train=n_per_label_train,
                                   n_per_label_test=n_per_label_test,
                                   store_root=store_root,
                                   classifier_store_path=classifier_store_path,
                                   gen_kwargs=gen_kwargs,
                                   use_existing_classifier=use_existing_classifier,
                                   verbose=verbose)
        return {"run_id": run_id, "mode": mode, "result": result}

    # MODE: zero_shot synthetic evaluation (no segmentation by default)
    elif mode == "zero_shot":
        if generate_images_fn is None or not labels or candidate_labels is None:
            raise ValueError("zero_shot requires generate_images_fn, labels and candidate_labels")
        # instantiate zero-shot manager with feature_extractor (for prototypes) and CLIP if provided
        zsm = ZeroShotManager(segmenter=None, feature_extractor=feat_inst,
                              clip_model=clip_model, clip_preprocess=clip_preprocess, device=clip_device)
        # if prototypes are already stored in store_root, user can pass them; otherwise ZeroShotManager will accept prototypes arg
        # here we load prototypes if exist
        prototypes = {}
        prot_root = os.path.join(store_root, "prototypes")
        if os.path.exists(prot_root):
            for lab in os.listdir(prot_root):
                ppth = os.path.join(prot_root, lab, "prototype.npy")
                if os.path.exists(ppth):
                    prototypes[lab] = np.load(ppth)
        result = zsm.evaluate(generate_images_fn=generate_images_fn,
                              labels=labels,
                              n_per_label_test=n_per_label_test,
                              candidate_labels=candidate_labels,
                              prototypes=prototypes if prototypes else None,
                              use_clip=use_clip and bool(clip_model),
                              use_prototypes=use_prototypes,
                              gen_kwargs=gen_kwargs)
        return {"run_id": run_id, "mode": mode, "result": result}

    # MODE: user_few_shot (segment user image then classify segments)
    elif mode == "user_few_shot":
        if image_path is None:
            raise ValueError("user_few_shot requires image_path")
        fsm = FewShotManager(feature_extractor=feat_inst, segmentation_model=seg_inst,
                             classifier_type=few_shot_classifier_type, classifier_params=classifier_params or {})
        result = fsm.classify_image(image_path=image_path,
                                    store_root=store_root,
                                    classifier_store_path=classifier_store_path,
                                    top_k=1,
                                    min_confidence=0.0)
        return {"run_id": run_id, "mode": mode, "result": result}

    # MODE: user_zero_shot (segment user image then zero-shot classify segments)
    elif mode == "user_zero_shot":
        if image_path is None or candidate_labels is None:
            raise ValueError("user_zero_shot requires image_path and candidate_labels")
        zsm = ZeroShotManager(segmenter=seg_inst, feature_extractor=feat_inst,
                              clip_model=clip_model, clip_preprocess=clip_preprocess, device=clip_device)
        result = zsm.classify_image(image_path=image_path,
                                    candidate_labels=candidate_labels,
                                    use_clip=use_clip and bool(clip_model),
                                    use_prototypes=use_prototypes,
                                    prototypes_store=store_root,
                                    top_k=1,
                                    min_confidence=0.0)
        return {"run_id": run_id, "mode": mode, "result": result}

    else:
        raise ValueError(f"Unknown mode '{mode}'")


if __name__ == "__main__":
    results = orchestrate(
        mode="zero_shot",
        generate_images_fn=generate_images,
        labels=["car", "phone"],
        n_per_label_test=10,
        candidate_labels=["car", "phone", "person", "cat", "jacket"],
        use_prototypes=True,
    )
    print("-------------- Zero Shot Results: -----------------------\n")
    print(results)
