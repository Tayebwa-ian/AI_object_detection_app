"""
Run orchestrator in few_shot mode using ResNet + prototype classifier.
"""

import pprint
from src.pipeline.orchestrator import orchestrate
from src.synthimage.generator import generate_images  # adjust import if needed

if __name__ == "__main__":
    results = orchestrate(
        mode="few_shot",
        segmentation_model_name=None,         # segmentation not needed in training/eval
        feature_extractor_name="resnet",
        few_shot_classifier_type="prototypes",
        generate_images_fn=generate_images,
        labels=["car", "phone"],
        n_per_label_train=5,
        n_per_label_test=3,
    )
    pprint.pprint(results)
