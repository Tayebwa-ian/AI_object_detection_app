"""
Run orchestrator in user_zero_shot mode:
Use DeepLab segmentation + EfficientNet features + CLIP.
"""

from .json_pretty_printer import save_json
import clip
from src.pipeline.orchestrator import orchestrate
from src.pipeline.config import DEVICE, DEFAULT_TEST_ROOT, CLIP_MODEL

if __name__ == "__main__":
    clip_model, clip_preprocess = clip.load(CLIP_MODEL, device=DEVICE)

    image_path = "src/pipeline/run/examples/image.png"   # replace with your image path

    results = orchestrate(
        mode="user_zero_shot",
        segmentation_model_name="deeplab",
        feature_extractor_name="efficientnet",
        labels=["cat"],
        candidate_labels=["dog", "cat", "car", "phone"],
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_device=DEVICE,
        image_path=image_path,
        use_clip=True,
        use_prototypes=True,
    )
    save_json(results, f"{DEFAULT_TEST_ROOT}/single_run/user/zeroshot_deeplab_efficient/results.json")
