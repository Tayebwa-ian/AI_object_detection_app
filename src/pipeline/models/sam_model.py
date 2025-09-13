"""
SAM wrapper module.

Loads a Segment Anything Model (SAM), runs automatic mask generation,
and reports inference metadata (inference_time, avg_confidence).
"""
import os
import urllib.request
import numpy as np
from typing import List, Dict, Any

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from ..config import SAM_CHECKPOINT, SAM_TYPE, DEVICE
from ..utils.timing import measure_time

class SAMWrapper:
    """
    Lightweight wrapper around the Segment Anything Model (SAM).

    Methods
    -------
    infer(image):
        Runs SAM automatic mask generation and returns (masks, metadata).
    """

    def __init__(self, checkpoint_path: str = None, model_type: str = None, device: str = None):
        """
        Initialize and load SAM model. Will download checkpoint if missing.

        Args:
            checkpoint_path: optional Path or str to checkpoint file. Defaults to config.SAM_CHECKPOINT
            model_type: optional model type name (e.g. "vit_h"). Defaults to config.SAM_TYPE
            device: torch device string. Defaults to config.DEVICE
        """
        import torch  # local import to keep module light
        self.checkpoint = str(checkpoint_path or SAM_CHECKPOINT)
        self.model_type = model_type or SAM_TYPE
        self.device = device or DEVICE

        if not os.path.exists(self.checkpoint):
            # Download from official source if not present
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{os.path.basename(self.checkpoint)}"
            urllib.request.urlretrieve(url, self.checkpoint)

        # Load model
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.model.to(device=self.device)
        self.generator = SamAutomaticMaskGenerator(self.model)

    @measure_time
    def infer(self, image) -> (List[Dict[str, Any]], Dict[str, Any]):
        """
        Generate segmentation masks for an image.

        Args:
            image: PIL.Image or np.ndarray

        Returns:
            masks: list of mask dicts produced by SAM
            metadata: dict containing avg_confidence (mean of predicted_iou) — inference_time added by decorator
        """
        image_np = np.array(image)
        masks = self.generator.generate(image_np)
        # Many masks include a 'predicted_iou' field in SAM outputs; fallback to 0 if absent
        confidences = [m.get("predicted_iou", 0.0) for m in masks]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        metadata = {"avg_confidence": avg_confidence, "num_masks": len(masks)}
        return masks, metadata
