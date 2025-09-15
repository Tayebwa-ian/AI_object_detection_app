"""
clip_feature.py

CLIP-based feature extractor wrapper that mimics the interface of ResNetWrapper.extract_features_from_path.

This is used as an alternative feature extractor in the pipeline.
"""
from typing import Dict, Any, Tuple
import time
import numpy as np
from PIL import Image

try:
    import clip
    import torch
except Exception:  # pragma: no cover
    clip = None
    torch = None

class CLIPFeatureExtractor:
    """
    Wrapper around OpenAI CLIP image encoder to extract image features.

    Usage:
        fe = CLIPFeatureExtractor(model_name="ViT-B/32", device="cpu")
        feat, meta = fe.extract_features_from_path("image.jpg")
    """
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        if clip is None:
            raise ImportError("clip package is not installed. Install via: pip install git+https://github.com/openai/CLIP.git")
        self.model_name = model_name
        self.device = device
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

    def _extract_image_tensor(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def extract_features_from_path(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract CLIP image features for a single image.

        Returns:
            feat: numpy ndarray (D,)
            meta: dict with model_name, device, feature_dim, inference_time
        """
        t0 = time.time()
        img_tensor = self._extract_image_tensor(image_path)
        with torch.no_grad():
            im_feat = self.model.encode_image(img_tensor)
            im_feat = im_feat.cpu().numpy().reshape(-1)
        elapsed = time.time() - t0
        # L2 normalize
        norm = np.linalg.norm(im_feat)
        if norm > 0:
            im_feat = im_feat / norm
        meta = {
            "model_name": f"CLIP-{self.model_name}",
            "device": self.device,
            "feature_dim": int(im_feat.shape[0]),
            "inference_time": float(elapsed)
        }
        return im_feat.astype(np.float32), meta
