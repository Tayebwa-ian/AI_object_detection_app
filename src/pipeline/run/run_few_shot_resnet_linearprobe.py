"""
Run orchestrator in few_shot mode using ResNet + linear probe classifier.
"""

from .json_pretty_printer import save_json
from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images
from src.pipeline.config import DEFAULT_TEST_ROOT

if __name__ == "__main__":
    results = orchestrate(
        mode="few_shot",
        segmentation_model_name=None,
        feature_extractor_name="resnet",
        few_shot_classifier_type="linear_probe",
        classifier_params={"epochs": 10, "lr": 1e-2, "batch_size": 16, "device": "cpu"},
        generate_images_fn=generate_images,
        labels=["apple", "banana"],
        n_per_label_train=8,
        n_per_label_test=4,
    )
    save_json(results, f"{DEFAULT_TEST_ROOT}/test/single_run/train/fewshot_resnet_linear_probe/results.json")

