"""
Plotting helpers for images and segmentation masks.

These are small wrappers around matplotlib to keep notebook-style plotting available.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def show_image(image, title: str = None):
    """
    Display a PIL or numpy image with no axes.

    Args:
        image: PIL.Image or numpy array
        title: optional title string
    """
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

def show_mask(mask, ax=None, random_color: bool = False):
    """
    Overlay a segmentation mask onto a matplotlib axis.

    Args:
        mask: boolean or {0,1} numpy array of shape (H, W)
        ax: matplotlib axis (if None, use current axis)
        random_color: use random RGBA color if True
    """
    if ax is None:
        ax = plt.gca()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks_on_image(raw_image, masks: List[Dict[str, Any]]):
    """
    Draws all masks (SAM-style dictionaries with 'segmentation') on the image.

    Args:
        raw_image: PIL/numpy image
        masks: list of dicts each containing "segmentation" as a 2D mask
    """
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    ax = plt.gca()
    for mask_data in masks:
        seg = mask_data.get("segmentation")
        if seg is not None:
            show_mask(np.array(seg), ax=ax, random_color=True)
    plt.axis("off")
    plt.show()
