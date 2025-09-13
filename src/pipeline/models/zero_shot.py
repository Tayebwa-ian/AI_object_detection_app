"""
Zero-shot classification utilities using CLIP and (optionally) prototype-free pipelines.

Two zero-shot strategies:
1. CLIP-based: uses a CLIP model to score text labels against an image.
2. ResNet-prototype: uses ResNet feature prototypes (already implemented in ResNetWrapper).

CLIP is optional — code will attempt to import it and raise a helpful error if missing.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from ..config import CLIP_MODEL, DEVICE
from ..utils.timing import measure_time

# Try to import OpenAI CLIP (the 'clip' package). If not installed, user must `pip install git+https://github.com/openai/CLIP.git`
try:
    import clip
except Exception as e:
    clip = None

class CLIPZeroShot:
    """
    Wrapper around CLIP zero-shot classification.

    Usage:
        z = CLIPZeroShot(model_name="ViT-B/32")
        result, meta = z.classify(image, candidate_labels=["cat","dog","car"])
    """

    def __init__(self, model_name: str = None, device: str = None):
        if clip is None:
            raise ImportError("Please install the 'clip' package: pip install git+https://github.com/openai/CLIP.git")
        self.device = device or DEVICE
        self.model_name = model_name or CLIP_MODEL
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

    @measure_time
    def classify(self, image, candidate_labels: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Classify an image in a zero-shot way using CLIP text prompts.

        Args:
            image: PIL image
            candidate_labels: list of label strings

        Returns:
            result: { 'labels': candidate_labels, 'probs': list of probabilities (aligned) , 'predicted': top_label }
            metadata: { 'avg_confidence': top_prob } (inference_time added by decorator)
        """
        # Preprocess image with clip preprocessing
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # 1 x C x H x W

        # Build text prompts and tokenize
        prompts = [f"a photo of a {label}" for label in candidate_labels]
        text_tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)  # 1 x D
            text_features = self.model.encode_text(text_tokens)    # N x D

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarities
            logits = (image_features @ text_features.T).cpu().numpy().reshape(-1)  # N
            # Convert to softmax probabilities
            probs = np.exp(logits) / np.exp(logits).sum()
            top_idx = int(probs.argmax())
            predicted = candidate_labels[top_idx]

        result = {"labels": candidate_labels, "probs": [float(p) for p in probs], "predicted": predicted}
        metadata = {"avg_confidence": float(max(probs))}

        return result, metadata
