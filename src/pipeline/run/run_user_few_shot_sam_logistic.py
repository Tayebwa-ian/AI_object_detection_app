"""
Run orchestrator in user_few_shot mode:
Use SAM segmentation + ResNet features + logistic classifier.
"""

from .json_pretty_printer import save_json
from src.pipeline.orchestrator import orchestrate
from src.pipeline.config import DEFAULT_TEST_ROOT

if __name__ == "__main__":
    image_path = "src/pipeline/run/examples/image.png"   # replace with your image path

    results = orchestrate(
        mode="user_few_shot",
        segmentation_model_name="sam",
        feature_extractor_name="resnet",
        few_shot_classifier_type="logistic",
    )
    save_json(results, f"{DEFAULT_TEST_ROOT}/single_run/user/fewshot_sam_resnet_logistic/results.json")
