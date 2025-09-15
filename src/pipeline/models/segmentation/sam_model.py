#!/usr/bin/python3"
"""
Improved SAMWrapper for image segmentation.

Enhancements:
- Preprocess images for better segmentation
- Post-process masks (NMS filtering, area filtering, confidence threshold)
- Save each segment as a unique PNG file inside a unique folder per input image
- Provide structured metadata for downstream tasks
"""
import os
import uuid
import urllib.request
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from ...config import SAM_CHECKPOINT, SAM_TYPE, DEVICE, BASE_DIR
from ...utils.timing import measure_time


class SAMWrapper:
    """
    Wrapper for Meta's Segment Anything Model (SAM).

    Features:
    - Improved segmentation through mask filtering and confidence thresholds
    - Automatic saving of each segment to disk with unique filenames
    - Metadata including inference time, resolution, avg_confidence
    """

    def __init__(self,
                 checkpoint_path: str = None,
                 model_type: str = None,
                 device: str = None,
                 min_area: int = 1000,
                 iou_threshold: float = 0.9,
                 conf_threshold: float = 0.8):
        """
        Initialize SAM model and mask generator.

        Args:
            checkpoint_path: optional checkpoint path (defaults to config.SAM_CHECKPOINT)
            model_type: SAM variant (defaults to config.SAM_TYPE)
            device: device string (defaults to config.DEVICE)
            min_area: minimum area of masks to keep (pixels)
            iou_threshold: IoU threshold for non-max suppression (remove overlapping masks)
            conf_threshold: minimum confidence (predicted_iou) to keep a mask
        """
        self.checkpoint = str(checkpoint_path or SAM_CHECKPOINT)
        self.model_type = model_type or SAM_TYPE
        self.device = device or DEVICE
        self.min_area = min_area
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        # Ensure checkpoint is available
        if not os.path.exists(self.checkpoint):
            url = f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, self.checkpoint)

        # Load model
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.model.to(device=self.device)
        self.generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=500,
        )

    def _nms_filter(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Non-Max Suppression (NMS) to filter overlapping masks.
        Keeps the highest-confidence mask in overlapping regions.

        Args:
            masks: list of mask dicts

        Returns:
            filtered list of masks
        """
        keep = []
        masks_sorted = sorted(masks, key=lambda m: m.get("predicted_iou", 0), reverse=True)

        def iou(m1, m2):
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            return inter / union if union > 0 else 0

        for mask in masks_sorted:
            seg = mask["segmentation"]
            if any(iou(seg, k["segmentation"]) > self.iou_threshold for k in keep):
                continue
            keep.append(mask)
        return keep

    def _filter_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter masks by confidence and area.
        """
        filtered = []
        for m in masks:
            conf = m.get("predicted_iou", 0)
            area = m.get("area", 0)
            if conf < self.conf_threshold:
                continue
            if area < self.min_area:
                continue
            filtered.append(m)
        return self._nms_filter(filtered)

    def _save_segments(self, masks: List[Dict[str, Any]], image: np.ndarray, base_folder: str) -> List[str]:
        """
        Save each mask as a separate PNG file.

        Args:
            masks: list of SAM masks
            image: numpy array of original image
            base_folder: folder to save segments

        Returns:
            list of saved file paths
        """
        os.makedirs(base_folder, exist_ok=True)
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        paths = []
        for idx, m in enumerate(masks_sorted[:10]):
            seg = m["segmentation"].astype(np.uint8) * 255
            seg_img = Image.fromarray(seg)
            filename = os.path.join(base_folder, f"segment_{idx:04d}.png")
            seg_img.save(filename)
            paths.append(filename)
        return paths

    @measure_time
    def segment_image(self, image_path: str, max_segments: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Segment an image and save results.

        Args:
            image_path: path to input image
            max_segments: maximum number of segments to save (optional)

        Returns:
            result: dict with 'segments' (paths), 'model_name', 'resolution'
            metadata: dict with 'num_segments', 'avg_confidence', 'inference_time'
        """
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        # Generate masks
        masks = self.generator.generate(img_np)

        # Filter masks
        masks = self._filter_masks(masks)

        # Optionally limit number of masks
        if max_segments:
            masks = masks[:max_segments]

        # Create unique folder for this image
        base_folder = BASE_DIR / f"segments/sam_{str(uuid.uuid4())}"
        seg_paths = self._save_segments(masks, img_np, base_folder)

        confidences = [m.get("predicted_iou", 0.0) for m in masks]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        result = {
            "segments": seg_paths,
            "model_name": f"SAM-{self.model_type.upper()}",
            "resolution": img.size  # (W, H)
        }
        metadata = {
            "num_segments": len(seg_paths),
            "avg_confidence": avg_conf
        }
        return result, metadata
