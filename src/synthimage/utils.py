#!/usr/bin/python3"
"""Utility helpers."""

import base64
import io
from PIL import Image
from typing import Tuple


def pil_to_base64_str(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Convert PIL Image to base64-encoded string (no data URI prefix).
    """
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return b64


def base64_str_to_pil(b64str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    """
    data = base64.b64decode(b64str)
    buffer = io.BytesIO(data)
    return Image.open(buffer).convert("RGBA")


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Return an RGB copy (preserve if already RGB)."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def center_crop_resize(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Center-crop the image to a square then resize to target_size (w, h).
    This keeps consistent framing for ResNet.
    """
    img = ensure_rgb(img)
    w, h = img.size
    # crop to square (min dimension)
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))
    img = img.resize(target_size, Image.LANCZOS)
    return img
