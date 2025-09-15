"""
efficientnet_wrapper.py

EfficientNet feature extractor wrapper.

This wrapper uses torchvision's EfficientNet (b0 by default) as a
feature extractor (global pooled features). It provides a consistent
interface:

    feat, meta = wrapper.extract_features_from_path(image_path)

Where:
 - feat is a 1-D numpy float32 vector (L2-normalized)
 - meta is a dict with keys: model_name, device, feature_dim, inference_time

Notes:
 - Requires torch and torchvision available.
 - Runs on CPU by default, but can use CUDA if available and device='cuda'.
"""

from typing import Tuple, Dict, Any
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models

class EfficientNetWrapper:
    """
    EfficientNet wrapper to extract image features.

    Args:
        model_name: one of "efficientnet_b0", "efficientnet_b1", ... (we default to b0)
        device: "cpu" or "cuda"
        pretrained: whether to load pretrained weights (default True)
    """
    def __init__(self, model_name: str = "efficientnet_b0", device: str = "cpu", pretrained: bool = True):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.pretrained = pretrained

        # instantiate model
        # uses torchvision models; available names: efficientnet_b0..b4 depending on torchvision version
        # we load model and remove classifier head to get feature vector
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not have '{model_name}'. Installed torchvision version may be old.")
        model_fn = getattr(models, model_name)
        self.model = model_fn(pretrained=pretrained)
        # remove classifier head to get penultimate features:
        # many torchvision EfficientNet implementations have .classifier; we will use features + pooling
        # Use the model up to the pooling/last layer: forward with features then adaptive avg pool if necessary
        self.model.eval()
        self.model.to(self.device)

        # Preprocess pipeline: ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

        # Attempt to determine output feature dim by doing a dummy forward
        with torch.no_grad():
            dummy = torch.zeros((1,3,224,224), device=self.device)
            # EfficientNet in torchvision forwards produce logits; but we want features before classifier
            # Many implementations expose .features or .avgpool; we try to detect a good hook
            try:
                # If model has 'features' and 'avgpool' attributes (common)
                if hasattr(self.model, "features"):
                    feats = self.model.features(dummy)
                    # some implementations have classifier after adaptive pooling
                    if hasattr(self.model, "avgpool"):
                        pooled = self.model.avgpool(feats)
                        flattened = torch.flatten(pooled, 1)
                        feat_dim = flattened.shape[1]
                    else:
                        # fallback: global average by adaptive avg pool
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1))
                        flattened = torch.flatten(pooled, 1)
                        feat_dim = flattened.shape[1]
                else:
                    # fallback: forward and remove classifier head dimension
                    out = self.model(dummy)
                    feat_dim = out.shape[1]
            except Exception:
                # final fallback
                out = self.model(dummy)
                feat_dim = out.shape[1]

        self.feature_dim = int(feat_dim)

    def _load_image_tensor(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        return tensor

    def extract_features_from_path(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract a feature vector for the given image path.

        Returns:
            feat (np.ndarray): 1-D float32 vector (L2-normalized)
            meta (dict): metadata including model_name, device, feature_dim, inference_time
        """
        t0 = time.time()
        img_tensor = self._load_image_tensor(image_path)
        with torch.no_grad():
            # robust extraction: try model.features + pooling, else forward and take output
            try:
                if hasattr(self.model, "features"):
                    feats = self.model.features(img_tensor)
                    if hasattr(self.model, "avgpool"):
                        pooled = self.model.avgpool(feats)
                    else:
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1))
                    flat = torch.flatten(pooled, 1)
                else:
                    # no features module: forward and take output; some nets already give features
                    flat = self.model(img_tensor)
            except Exception:
                # last resort: forward
                flat = self.model(img_tensor)
            feat = flat.cpu().numpy().reshape(-1)
        elapsed = time.time() - t0

        # L2 normalization
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        meta = {
            "model_name": self.model_name,
            "device": self.device,
            "feature_dim": int(self.feature_dim),
            "inference_time": float(elapsed)
        }
        print(f"[EfficientNetWrapper] extracted feature dim={meta['feature_dim']} time={meta['inference_time']:.4f}s for {image_path}")
        return feat.astype(np.float32), meta
