"""
Run orchestrator in few_shot mode using ResNet + logistic classifier.
"""

from .json_pretty_printer import save_json
from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images  # adjust import if needed
from ..config import DEFAULT_TEST_ROOT

if __name__ == "__main__":
    results = orchestrate(
        mode="few_shot",
        segmentation_model_name=None,         # segmentation not needed in training/eval
        feature_extractor_name="resnet",
        few_shot_classifier_type="logistic",
        generate_images_fn=generate_images,
        labels=["cat", "car", "phone"],
        n_per_label_train=20,
        n_per_label_test=5,
    )
    save_json(results, f"{DEFAULT_TEST_ROOT}/single_run/train/fewshot_resnet_logistic/results.json")
