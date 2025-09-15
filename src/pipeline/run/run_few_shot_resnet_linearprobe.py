"""
Run orchestrator in few_shot mode using ResNet + linear probe classifier.
"""

import pprint
from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images
from src.pipeline.config import DEFAULT_OUTPUT_ROOT, LINEAR_SVC_CLASSIFIER_PATH

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
        store_root=DEFAULT_OUTPUT_ROOT,
        classifier_store_path=LINEAR_SVC_CLASSIFIER_PATH,
    )
    pprint.pprint(results)
