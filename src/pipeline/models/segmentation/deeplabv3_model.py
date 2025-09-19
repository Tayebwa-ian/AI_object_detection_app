"""
DeepLabv3+ Wrapper for semantic segmentation.

Features:
- Pretrained DeepLabv3+ (ResNet-101 backbone from torchvision)
- Semantic segmentation of input images
- Saving each class segment as separate PNG masks in a unique folder
- Metadata with inference time, resolution, model name, avg_confidence, num_segments
"""
import os
import uuid
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation

from ...config import DEVICE, IMAGE_SIZE, BASE_DIR
from ...utils.timing import measure_time


class DeepLabV3Wrapper:
    """
    Wrapper for torchvision DeepLabv3+ model (ResNet-101 backbone).

    Produces semantic segmentation maps, saves each class as a mask,
    and returns metadata similar to SAMWrapper.
    """

    def __init__(self,
                 pretrained: bool = True,
                 device: str = None,
                 min_area: int = 1000,
                 conf_threshold: float = 0.5):
        """
        Initialize DeepLabv3+ model.

        Args:
            pretrained: whether to use pretrained weights (on COCO or VOC dataset)
            device: device string (default: config.DEVICE)
            min_area: minimum pixel area to keep a mask
            conf_threshold: probability threshold to keep a mask
        """
        self.device = device or DEVICE
        self.min_area = min_area
        self.conf_threshold = conf_threshold

        # Load pretrained DeepLabv3+ with ResNet-101 backbone
        self.model = segmentation.deeplabv3_resnet101(pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _save_segments(self, masks: Dict[int, np.ndarray], base_folder: str) -> List[str]:
        """
        Save segmentation masks per class as PNG files.

        Args:
            masks: dict {class_id: binary_mask}
            base_folder: directory to save masks

        Returns:
            list of saved mask file paths
        """
        os.makedirs(base_folder, exist_ok=True)
        paths = []
        for idx, (cls, mask) in enumerate(masks.items(), 1):
            seg = (mask.astype(np.uint8) * 255)
            seg_img = Image.fromarray(seg)
            filename = os.path.join(base_folder, f"class_{cls:02d}_segment_{idx:04d}.png")
            seg_img.save(filename)
            paths.append(filename)
        return paths

    @measure_time
    def segment_image(self, image_path: str, max_segments: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform semantic segmentation on an image.

        Args:
            image_path: path to input image
            max_segments: maximum number of classes to save (optional)

        Returns:
            result: dict with 'segments', 'model_name', 'resolution'
            metadata: dict with 'num_segments', 'avg_confidence', 'inference_time'
        """
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)["out"]  # shape [1, C, H, W]
            probs = F.softmax(output, dim=1)[0]  # [C, H, W]

        # Resize to original resolution
        probs_resized = F.interpolate(probs.unsqueeze(0),
                                      size=(orig_h, orig_w),
                                      mode="bilinear",
                                      align_corners=False)[0]

        # For each class, generate mask if above threshold
        masks = {}
        confidences = []
        for cls in range(probs_resized.shape[0]):
            prob_map = probs_resized[cls].cpu().numpy()
            mask = prob_map > self.conf_threshold
            area = mask.sum()
            if area < self.min_area:
                continue
            masks[cls] = mask
            confidences.append(prob_map[mask].mean())

        # Limit number of saved segments
        if max_segments:
            # Sort by confidence, keep top-k
            sorted_masks = sorted(masks.items(),
                                  key=lambda kv: -np.mean(probs_resized[kv[0]].cpu().numpy()[kv[1]]))
            masks = dict(sorted_masks[:max_segments])

        # Create unique folder
        base_folder = BASE_DIR / f"src/pipeline/store/segments/deeplab_{str(uuid.uuid4())}"
        seg_paths = self._save_segments(masks, base_folder)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        result = {
            "segments": seg_paths,
            "model_name": "DeepLabv3+ (ResNet-101)",
            "resolution": (orig_w, orig_h)
        }
        metadata = {
            "num_segments": len(seg_paths),
            "avg_confidence": avg_conf
        }
        return result, metadata
