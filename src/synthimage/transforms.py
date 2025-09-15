#!/usr/bin/python3"
"""
Image augmentation utilities.
These functions vary object clarity vs. background clarity in a simple but effective way,
while preserving overall image quality.
"""

import random
from typing import Tuple, Optional
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw
import numpy as np
from .utils import ensure_rgb, center_crop_resize


class ImageAugmentor:
    """
    Encapsulates randomized augmentations.

    Parameters
    ----------
    target_size: (w, h)
        final output size (e.g. (224, 224))
    seed: Optional[int]
        Random seed for reproducibility
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224), seed: Optional[int] = None):
        self.target_size = target_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def apply_random_transforms(self, img: Image.Image) -> Image.Image:
        """
        Apply a random sequence of transformations:
        - random flip
        - random rotate
        - random gaussian blur (whole image or background)
        - random gaussian noise
        - random resolution downscale/upscale
        - random contrast/brightness adjustments
        - final sharpening (to preserve clarity)

        The final image is resized to self.target_size using center-crop-resize.
        """
        img = ensure_rgb(img)

        # Random flip
        if random.random() < 0.2:
            img = ImageOps.mirror(img)

        # Random rotation
        if random.random() < 0.4:
            angle = random.uniform(-25, 25)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(127,127,127))

        # Decide whether to blur background or whole image
        if random.random() < 0.3:
            img = self._blur_background_with_center_object(img)
        else:
            if random.random() < 0.3:
                radius = random.uniform(0.5, 2.5)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Add gaussian noise
        if random.random() < 0.3:
            img = self._add_gaussian_noise(img, sigma=random.uniform(3, 15))

        # Random resolution change
        if random.random() < 0.2:
            downscale = random.choice([0.5, 0.6, 0.75, 0.9])
            w, h = img.size
            new_w, new_h = max(1, int(w * downscale)), max(1, int(h * downscale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img = img.resize((w, h), Image.LANCZOS)

        # Random brightness/contrast jitter
        if random.random() < 0.2:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.85, 1.2))
        if random.random() < 0.2:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.15))

        # Final crop & resize
        img = center_crop_resize(img, self.target_size)

        # Final sharpening step for clarity
        if random.random() < 0.5:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))

        return img

    def _blur_background_with_center_object(self, img: Image.Image) -> Image.Image:
        """
        Heuristic: assume the main object is near the center.
        Create an elliptical mask for the 'object' (kept sharp) and blur the background.
        """
        img = ensure_rgb(img)
        w, h = img.size

        # object occupies random proportion of image center
        prop = random.uniform(0.25, 0.6)
        ow, oh = int(w * prop), int(h * prop)
        left, top = (w - ow) // 2, (h - oh) // 2
        right, bottom = left + ow, top + oh

        # blurred background
        blur_radius = random.uniform(2.0, 5.0)
        bg = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # object region
        obj = img.crop((left, top, right, bottom))
        if random.random() < 0.4:
            obj = obj.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # elliptical mask with feathering
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([left, top, right, bottom], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

        # composite
        composite = Image.composite(img, bg, mask)
        return composite

    def _add_gaussian_noise(self, img: Image.Image, sigma: float = 10.0) -> Image.Image:
        """
        Add zero-mean gaussian noise with standard deviation sigma (0-255 scale).
        """
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        noisy = arr + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
