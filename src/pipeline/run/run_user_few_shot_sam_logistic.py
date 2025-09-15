"""
Run orchestrator in user_few_shot mode:
Use SAM segmentation + ResNet features + logistic classifier.
"""

import pprint
from src.pipeline.orchestrator import orchestrate
from src.pipeline.config import LOGISTIC_CLASSIFIER_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_CLASSIFIER_PATH

if __name__ == "__main__":
    image_path = "src/pipeline/run/examples/image.png"   # replace with your image path

    results = orchestrate(
        mode="user_few_shot",
        segmentation_model_name="sam",
        feature_extractor_name="resnet",
        few_shot_classifier_type="logistic",
        store_root=DEFAULT_OUTPUT_ROOT,
        classifier_store_path=DEFAULT_CLASSIFIER_PATH,
        image_path=image_path,
    )
    pprint.pprint(results)
