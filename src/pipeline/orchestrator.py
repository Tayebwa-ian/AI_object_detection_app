#!/usr/bin/python3
"""
orchestrator.py

Unified orchestration for few-shot and zero-shot workflows.

Improvements:
-------------
- Allow passing constructor arguments to FewShotManager and ZeroShotManager.
  For example, you can set custom ResNetWrapper/SAMWrapper, classifier_class,
  CLIP model name, or device.
- Synthetic workflows (few_shot, zero_shot) do NOT use segmentation.
- User workflows (user_few_shot, user_zero_shot) use segmentation.

Modes:
------
- "few_shot": run synthetic training + evaluation with FewShotManager (no segmentation).
- "zero_shot": run synthetic evaluation with ZeroShotManager (no segmentation).
- "user_few_shot": classify a real user image with FewShotManager (segmentation applied).
- "user_zero_shot": classify a real user image with ZeroShotManager (segmentation applied).
"""

from typing import Callable, Dict, Any, List, Optional

from .models.workflows.few_shot_manager import FewShotManager
from .models.workflows.zero_shot_manager import ZeroShotManager
from .config import DEFAULT_CLASSIFIER_PATH, DEFAULT_OUTPUT_ROOT, DEVICE, CLIP_MODEL
from ..synthimage.generator import generate_images


# ----------------------------
# Few-shot synthetic workflow
# ----------------------------
def run_few_shot_workflow(
    generate_images_fn: Callable[..., List],
    labels: List[str],
    n_per_label_train: int,
    n_per_label_test: int,
    store_root: str,
    classifier_path: str,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    use_existing_classifier: bool = False,
    classifier_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    # NEW: pass constructor args
    few_shot_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run few-shot synthetic workflow (no segmentation).
    """
    fsm = FewShotManager(**(few_shot_kwargs or {}))
    return fsm.test_pipeline(
        generate_images_fn=generate_images_fn,
        labels=labels,
        n_per_label_train=n_per_label_train,
        n_per_label_test=n_per_label_test,
        store_root=store_root,
        classifier_path=classifier_path,
        classifier_params=classifier_params,
        gen_kwargs=gen_kwargs,
        use_existing_classifier=use_existing_classifier,
        verbose=verbose
    )


# ----------------------------
# Zero-shot synthetic workflow
# ----------------------------
def run_zero_shot_workflow(
    generate_images_fn: Callable[..., List],
    labels: List[str],
    n_per_label_test: int,
    candidate_labels: List[str],
    use_clip: bool = True,
    use_prototypes: bool = True,
    prototypes_store: Optional[str] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    # NEW: pass constructor args
    zero_shot_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run zero-shot synthetic workflow (no segmentation).
    """
    zsm = ZeroShotManager(**(zero_shot_kwargs or {}))
    if verbose:
        print("[orchestrator] Running zero-shot workflow...")
    return zsm.evaluate(
        generate_images_fn=generate_images_fn,
        labels=labels,
        n_per_label_test=n_per_label_test,
        candidate_labels=candidate_labels,
        prototypes_store=prototypes_store,
        use_clip=use_clip,
        use_prototypes=use_prototypes,
        gen_kwargs=gen_kwargs
    )


# ----------------------------
# User workflows (segmentation)
# ----------------------------
def run_user_few_shot_workflow(
    image_path: str,
    store_root: str,
    classifier_path: Optional[str] = None,
    # NEW: pass constructor args
    few_shot_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify a real user-submitted image using FewShotManager.
    Segmentation is applied.
    """
    fsm = FewShotManager(**(few_shot_kwargs or {}))
    return fsm.classify_image(
        image_path=image_path,
        store_root=store_root,
        classifier_path=classifier_path
    )


def run_user_zero_shot_workflow(
    image_path: str,
    candidate_labels: List[str],
    prototypes_store: Optional[str] = None,
    use_clip: bool = True,
    use_prototypes: bool = True,
    # NEW: pass constructor args
    zero_shot_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify a real user-submitted image using ZeroShotManager.
    Segmentation is applied.
    """
    zsm = ZeroShotManager(**(zero_shot_kwargs or {}))
    return zsm.classify_image(
        image_path=image_path,
        candidate_labels=candidate_labels,
        prototypes_store=prototypes_store,
        use_clip=use_clip,
        use_prototypes=use_prototypes
    )


# ----------------------------
# Unified orchestrator
# ----------------------------
def orchestrate(
    mode: str,
    # common args
    generate_images_fn: Optional[Callable[..., List]] = None,
    labels: Optional[List[str]] = None,
    n_per_label_train: Optional[int] = None,
    n_per_label_test: int = 10,
    candidate_labels: Optional[List[str]] = None,
    store_root: str = DEFAULT_OUTPUT_ROOT,
    classifier_path: str = DEFAULT_CLASSIFIER_PATH,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    use_existing_classifier: bool = False,
    classifier_params: Optional[Dict[str, Any]] = None,
    use_clip: bool = True,
    use_prototypes: bool = True,
    # user mode args
    image_path: Optional[str] = None,
    # NEW: pass constructor args
    few_shot_kwargs: Optional[Dict[str, Any]] = None,
    zero_shot_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Unified orchestrator entry point.

    Modes:
        - "few_shot": synthetic training/testing (no segmentation)
        - "zero_shot": synthetic evaluation (no segmentation)
        - "user_few_shot": classify a user image (segmentation applied)
        - "user_zero_shot": classify a user image (segmentation applied)
    """
    if mode == "few_shot":
        if not (generate_images_fn and labels and n_per_label_train is not None):
            raise ValueError("few_shot requires generate_images_fn, labels, n_per_label_train.")
        return run_few_shot_workflow(
            generate_images_fn=generate_images_fn,
            labels=labels,
            n_per_label_train=n_per_label_train,
            n_per_label_test=n_per_label_test,
            store_root=store_root,
            classifier_path=classifier_path,
            gen_kwargs=gen_kwargs,
            use_existing_classifier=use_existing_classifier,
            classifier_params=classifier_params,
            few_shot_kwargs=few_shot_kwargs,
            verbose=verbose
        )
    elif mode == "zero_shot":
        if not (generate_images_fn and labels and candidate_labels):
            raise ValueError("zero_shot requires generate_images_fn, labels, candidate_labels.")
        return run_zero_shot_workflow(
            generate_images_fn=generate_images_fn,
            labels=labels,
            n_per_label_test=n_per_label_test,
            candidate_labels=candidate_labels,
            use_clip=use_clip,
            use_prototypes=use_prototypes,
            prototypes_store=store_root,
            gen_kwargs=gen_kwargs,
            zero_shot_kwargs=zero_shot_kwargs,
            verbose=verbose
        )
    elif mode == "user_few_shot":
        if not image_path:
            raise ValueError("user_few_shot requires image_path.")
        return run_user_few_shot_workflow(
            image_path=image_path,
            store_root=store_root,
            classifier_path=classifier_path,
            few_shot_kwargs=few_shot_kwargs
        )
    elif mode == "user_zero_shot":
        if not (image_path and candidate_labels):
            raise ValueError("user_zero_shot requires image_path and candidate_labels.")
        return run_user_zero_shot_workflow(
            image_path=image_path,
            candidate_labels=candidate_labels,
            prototypes_store=store_root,
            use_clip=use_clip,
            use_prototypes=use_prototypes,
            zero_shot_kwargs=zero_shot_kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: 'few_shot', 'zero_shot', 'user_few_shot', 'user_zero_shot'.")


if __name__ == "__main__":
    # Example: run zero-shot with CLIP only
    results = orchestrate(
        mode="zero_shot",
        generate_images_fn=generate_images,
        labels=["car", "phone"],
        n_per_label_test=6,
        candidate_labels=["car", "phone", "person"],
        use_prototypes=False,
        use_clip=True,
        zero_shot_kwargs={"sam": None, "clip_model_name": CLIP_MODEL, "device": DEVICE},
        verbose=True
    )
    print("-------------- Zero Shot Results: -----------------------\n")
    print(results["metrics"])
