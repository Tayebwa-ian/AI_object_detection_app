"""
Run orchestrator in few_shot mode using ResNet + logistic classifier.
"""

import pprint
from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images  # adjust import if needed

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
    pprint.pprint(results)
