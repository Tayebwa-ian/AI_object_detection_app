"""
Run orchestrator in few_shot mode using EfficientNet + logistic regression classifier.
"""

from .json_pretty_printer import save_json
from src.pipeline.orchestrator import orchestrate
from synthimage.generator import generate_images
from src.pipeline.config import DEFAULT_TEST_ROOT

if __name__ == "__main__":
    results = orchestrate(
        mode="few_shot",
        segmentation_model_name=None,
        feature_extractor_name="efficientnet",
        few_shot_classifier_type="logistic",
        classifier_params={"max_iter": 200},
        generate_images_fn=generate_images,
        labels=["dog", "cat"],
        n_per_label_train=10,
        n_per_label_test=4,
    )
    save_json(results, f"{DEFAULT_TEST_ROOT}/test/single_run/train/fewshot_efficientnet_logistic/results.json")
