"""
Run orchestrator in few_shot mode using EfficientNet + logistic regression classifier.
"""

import pprint
from src.pipeline.orchestrator import orchestrate
from synthimage.generator import generate_images
from src.pipeline.config import DEFAULT_OUTPUT_ROOT, LOGISTIC_CLASSIFIER_PATH

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
        store_root=DEFAULT_OUTPUT_ROOT,
        classifier_store_path=LOGISTIC_CLASSIFIER_PATH,
    )
    pprint.pprint(results)
