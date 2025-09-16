#!/usr/bin/python3
"""
Unified runner to test orchestrator with multiple modes and model combinations.

Covers:
- Few-shot with prototypes, logistic regression, linear probe
- Zero-shot with CLIP and/or prototypes
- User image workflows (few-shot & zero-shot) with segmentation applied

Results are printed and stored in `output/runs/<run_id>/`.
"""

import pprint
import uuid
from .json_pretty_printer import save_json, ensure_dir

import clip

from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images
from src.pipeline.config import DEVICE, CLIP_MODEL, DEFAULT_TEST_ROOT


# -------------------------
# Helpers
# -------------------------

def get_clip():
    """Lazy load CLIP with device detection."""
    device = DEVICE
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    return model, preprocess, device


# -------------------------
# Configs
# -------------------------
FEW_SHOT_CLASSIFIERS = ["prototypes", "logistic", "linear_probe"]
FEATURE_EXTRACTORS = ["resnet", "efficientnet"]
SEGMENTATION_MODELS = [None, "sam", "deeplab"]
MODES = ["few_shot", "zero_shot", "user_few_shot", "user_zero_shot"]

# Candidate labels for zero-shot tests
CANDIDATE_LABELS = ["car", "phone", "dog", "cat", "jacket"]


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_id = str(uuid.uuid4())[:8]
    out_root = DEFAULT_TEST_ROOT
    ensure_dir(out_root)

    print(f"=== Starting orchestrator unified test run: {run_id} ===")

    # Load CLIP once for zero-shot tests
    clip_model, clip_preprocess, clip_device = get_clip()

    for mode in MODES:
        print(f"\n--- Mode: {mode} ---")

        for feat_extractor in FEATURE_EXTRACTORS:
            for seg_model in SEGMENTATION_MODELS:
                # skip segmentation when not needed
                if seg_model is not None and mode in ["few_shot", "zero_shot"]:
                    continue

                # few-shot branch
                if mode == "few_shot":
                    for clf in FEW_SHOT_CLASSIFIERS:
                        print(f"Running {mode} with {feat_extractor} + {clf}")
                        result = orchestrate(
                            mode=mode,
                            segmentation_model_name=seg_model,
                            feature_extractor_name=feat_extractor,
                            few_shot_classifier_type=clf,
                            generate_images_fn=generate_images,
                            labels=["car", "phone"],
                            n_per_label_train=4,
                            n_per_label_test=3,
                            store_root=f"{out_root}/{mode}_{feat_extractor}_{clf}",
                            classifier_store_path=f"{out_root}/{mode}_{feat_extractor}_{clf}/clf.joblib",
                            classifier_params={"max_iter": 200} if clf == "logistic" else None,
                            verbose=False,
                        )
                        save_json(result, f"{out_root}/{mode}_{feat_extractor}_{clf}/results.json")

                # zero-shot branch
                elif mode == "zero_shot":
                    print(f"Running {mode} with {feat_extractor} (clip+proto)")
                    result = orchestrate(
                        mode=mode,
                        segmentation_model_name=seg_model,
                        feature_extractor_name=feat_extractor,
                        generate_images_fn=generate_images,
                        labels=["car", "phone"],
                        n_per_label_test=3,
                        candidate_labels=CANDIDATE_LABELS,
                        use_clip=True,
                        use_prototypes=True,
                        clip_model=clip_model,
                        clip_preprocess=clip_preprocess,
                        clip_device=clip_device,
                        store_root=f"{out_root}/{mode}_{feat_extractor}",
                        verbose=False,
                    )
                    save_json(result, f"{out_root}/{mode}_{feat_extractor}/results.json")

                # user image workflows (simulate with example images)
                elif mode == "user_few_shot":
                    if seg_model is None:
                        continue
                    for clf in FEW_SHOT_CLASSIFIERS:
                        print(f"Running {mode} with {seg_model} + {feat_extractor} + {clf}")
                        result = orchestrate(
                            mode=mode,
                            segmentation_model_name=seg_model,
                            feature_extractor_name=feat_extractor,
                            few_shot_classifier_type=clf,
                            image_path="src/pipeline/run/examples/image.png",
                            store_root=f"{out_root}/{mode}_{seg_model}_{feat_extractor}_{clf}",
                            classifier_store_path=f"{out_root}/{mode}_{seg_model}_{feat_extractor}_{clf}/clf.joblib",
                            verbose=False,
                        )
                        save_json(result, f"{out_root}/{mode}_{seg_model}_{feat_extractor}_{clf}/results.json")

                elif mode == "user_zero_shot":
                    if seg_model is None:
                        continue
                    print(f"Running {mode} with {seg_model} + {feat_extractor}")
                    result = orchestrate(
                        mode=mode,
                        segmentation_model_name=seg_model,
                        feature_extractor_name=feat_extractor,
                        candidate_labels=CANDIDATE_LABELS,
                        image_path="src/pipeline/run/examples/image.png",
                        use_clip=True,
                        use_prototypes=True,
                        clip_model=clip_model,
                        clip_preprocess=clip_preprocess,
                        clip_device=clip_device,
                        store_root=f"{out_root}/{mode}_{seg_model}_{feat_extractor}",
                        verbose=False,
                    )
                    save_json(result, f"{out_root}/{mode}_{seg_model}_{feat_extractor}/results.json")

    print(f"\n=== Run finished. Results stored under {out_root} ===")
