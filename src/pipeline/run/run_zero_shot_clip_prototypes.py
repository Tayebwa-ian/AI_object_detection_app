"""
Run orchestrator in zero_shot mode using CLIP + prototypes.
"""

import pprint
import clip
from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images
from src.pipeline.config import CLIP_MODEL, DEFAULT_OUTPUT_ROOT, DEVICE


if __name__ == "__main__":
    clip_model, clip_preprocess = clip.load(CLIP_MODEL, device=DEVICE)

    results = orchestrate(
        mode="zero_shot",
        segmentation_model_name=None,      # no segmentation needed
        feature_extractor_name="resnet",
        candidate_labels=["car", "phone", "dog", "cat"],
        generate_images_fn=generate_images,
        labels=["car", "phone"],
        n_per_label_test=5,
        store_root=DEFAULT_OUTPUT_ROOT,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_device=DEVICE,
        use_clip=True,
        use_prototypes=True,
    )
    pprint.pprint(results)
