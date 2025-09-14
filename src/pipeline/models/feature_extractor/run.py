"""
Feature pipeline orchestrator using ResNetWrapper.

Provides a single entry function `run_feature_pipeline` that performs
feature extraction, prototype storage, finalization, or classification
based on simple arguments.

This allows quick usage of the ResNetWrapper without handling multiple steps.
"""
from typing import Dict, Any, Optional, List, Union

from .resnet_model import ResNetWrapper
from ...config import DEFAULT_OUTPUT_ROOT


def run_feature_pipeline(
    task: str,
    store_root: str = DEFAULT_OUTPUT_ROOT,
    image_path: Optional[Union[str, List[str]]] = None,
    label: Optional[str] = None,
    model_name: str = "resnet50",
    pretrained: bool = True,
    tta: bool = True,
    top_k: int = 1
) -> Dict[str, Any]:
    """
    Run a feature/prototype pipeline task with minimal arguments.

    Args:
        task (str): One of {"extract", "add", "finalize", "classify", "list"}.
        store_root (str): Root directory for storing samples/prototypes.
        image_path (str or list): Path(s) to image(s), required for 'extract', 'add', 'classify'.
        label (str): Label name, required for 'add'.
        model_name (str): ResNet backbone name (default "resnet50").
        pretrained (bool): Load pretrained weights.
        tta (bool): Apply test-time augmentation during feature extraction.
        top_k (int): Number of top predictions to return (for classification).

    Returns:
        dict containing results and metadata depending on task.
    """
    wrapper = ResNetWrapper(model_name=model_name, pretrained=pretrained, tta=tta)

    if task == "extract":
        if not image_path:
            raise ValueError("image_path required for task=extract")
        feat, meta = wrapper.extract_features_from_path(image_path)
        return {"features": feat.tolist(), "metadata": meta}

    elif task == "add":
        if not image_path or not label:
            raise ValueError("image_path and label required for task=add")
        if isinstance(image_path, str):
            image_path = [image_path]
        added = []
        for p in image_path:
            added.append(wrapper.add_prototype(label=label, image_path=p, store_root=store_root))
        return {"added_samples": added}

    elif task == "finalize":
        protos = wrapper.finalize_prototypes(store_root)
        return {"prototypes": protos}

    elif task == "classify":
        if not image_path:
            raise ValueError("image_path required for task=classify")
        result, meta = wrapper.classify_with_prototypes(image_path=image_path, store_root=store_root, top_k=top_k)
        return {"result": result, "metadata": meta}

    elif task == "list":
        idx = wrapper.list_prototypes(store_root)
        return {"index": idx}

    else:
        raise ValueError(f"Unsupported task: {task}. Use extract/add/finalize/classify/list.")
