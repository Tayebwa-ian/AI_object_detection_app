#!/usr/bin/python3"
"""
examples/generate_and_save.py

Example script demonstrating how to use the synthimage package to:
 - call the AI image generation endpoint using the updated payload shape
 - request images using the same generation parameters you used before
 - optionally provide an example image
 - apply randomized augmentations
 - display and save generated images into label folders

This file is intended to replace the previous example and to call the
ImageGenClient in the same way your working script did (model, sampling,
cfg_scale, guidance, strength, etc.).

Usage (recommended):
  1. Set environment variables:
       export SYNTHIMAGE_API_TOKEN="<your_token>"
       export SYNTHIMAGE_API_ENDPOINT="https://llm-web.aieng.fim.uni-passau.de"
     (You may also edit the constants below.)
  2. Run:
       python examples/generate_and_save.py

The script is intentionally simple and well-documented. Adjust parameters
below to suit your dataset generation needs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List

from PIL import Image
from .transforms import ImageAugmentor
from .api_client import ImageGenClient
from .storage import StorageManager
from .viewer import Viewer


# ------------------------------- Configuration -------------------------------
API_ENDPOINT = os.environ.get(
    "SYNTHIMAGE_API_ENDPOINT",
    "https://llm-web.aieng.fim.uni-passau.de",
)
API_TOKEN = os.environ.get("SYNTHIMAGE_API_TOKEN", "gpustack_adf7d482bd8a814b_a1bfc829fc58b64de0d65cdd91473815")

DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_MODEL = "flux.1-schnell-gguf"
DEFAULT_SAMPLING_STEPS = 20
DEFAULT_SAMPLE_METHOD = "euler"
DEFAULT_CFG_SCALE = 1.0
DEFAULT_GUIDANCE = 3.5
DEFAULT_STRENGTH = 0.75

DEFAULT_LABEL = "jacket"
DEFAULT_NUM_IMAGES = 20
DEFAULT_OUTPUT_ROOT = "src/synthimage/generated_images"
DEFAULT_EXAMPLE_IMAGE_PATH: Optional[str] = None

MAX_BATCH_SIZE = 4

# ------------------------------- Logging setup -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("examples.generate_and_save")


def generate_in_batches(client: ImageGenClient, prompt: str, example_image: Optional[Image.Image], total: int,
                        model: str, sampling_steps: int, sample_method: str,
                        cfg_scale: float, guidance: float, strength: float) -> List[Image.Image]:
    """
    Generate images in batches of MAX_BATCH_SIZE until reaching `total`.
    """
    results: List[Image.Image] = []
    remaining = total

    while remaining > 0:
        n = min(MAX_BATCH_SIZE, remaining)
        logger.info("Requesting batch of %d images (remaining=%d)", n, remaining)

        batch = client.generate(
            label=prompt,
            example_image=example_image,
            n=n,
            size=(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            model=model,
            sampling_steps=sampling_steps,
            sample_method=sample_method,
            cfg_scale=cfg_scale,
            guidance=guidance,
            strength=strength,
            extra_payload={
                "schedule_method": "discrete",
                "negative_prompt": "",
                "seed": None,
            },
        )
        results.extend(batch)
        remaining -= n

    return results


def generate_images(
    num_images: int = DEFAULT_NUM_IMAGES,
    label: str = DEFAULT_LABEL,
    model: str = DEFAULT_MODEL,
    sampling_steps: int = DEFAULT_SAMPLING_STEPS,
    sample_method: str = DEFAULT_SAMPLE_METHOD,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    guidance: float = DEFAULT_GUIDANCE,
    strength: float = DEFAULT_STRENGTH,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    example_image_path: Optional[str] = DEFAULT_EXAMPLE_IMAGE_PATH,
) -> List[Path]:
    """
    Generate, augment, and save images with given parameters.

    Parameters
    ----------
    num_images : int
        Total number of images to generate.
    label : str
        Object label for naming and folder storage.
    model : str
        Model name to use for generation.
    sampling_steps : int
        Number of sampling steps.
    sample_method : str
        Sampling method (e.g. "euler").
    cfg_scale : float
        CFG scale value.
    guidance : float
        Guidance strength.
    strength : float
        Strength parameter.
    output_root : str
        Root directory where generated images will be saved. A subfolder with
        the given label is created inside this root.
    example_image_path : Optional[str]
        Optional path to an example conditioning image.

    Returns
    -------
    List[Path]
        List of saved image file paths.
    """
    if not API_TOKEN or API_TOKEN.startswith("<PUT_"):
        raise ValueError("API token is not set. Please set SYNTHIMAGE_API_TOKEN environment variable.")

    client = ImageGenClient(
        endpoint=API_ENDPOINT,
        token=API_TOKEN,
        default_size=(DEFAULT_WIDTH, DEFAULT_HEIGHT),
    )

    augmentor = ImageAugmentor(target_size=(DEFAULT_WIDTH, DEFAULT_HEIGHT), seed=None)
    storage = StorageManager(root_dir=output_root, output_format="PNG")

    example_image = None
    if example_image_path:
        p = Path(example_image_path)
        if p.exists():
            example_image = Image.open(p).convert("RGB")
            logger.info("Loaded example image from %s", p)
        else:
            logger.warning("EXAMPLE_IMAGE_PATH specified but file does not exist: %s", p)

    prompt = f"A clear photo of a {label}"

    images = generate_in_batches(
        client, prompt, example_image, num_images,
        model, sampling_steps, sample_method, cfg_scale, guidance, strength
    )

    saved_paths: List[Path] = []
    for idx, img in enumerate(images, start=1):
        augmented = augmentor.apply_random_transforms(img)
        Viewer.display(augmented)
        filename = f"{label}_{idx:04d}"
        saved_path = storage.save_image(augmented, label=label, filename=filename)
        logger.info("Saved image to %s", saved_path)
        saved_paths.append(saved_path)

    return saved_paths


def main() -> None:
    """Main entrypoint for the example script."""
    try:
        saved = generate_images(num_images=DEFAULT_NUM_IMAGES, label=DEFAULT_LABEL, output_root=DEFAULT_OUTPUT_ROOT)
        logger.info("Done. Generated %d images under '%s'", len(saved), DEFAULT_OUTPUT_ROOT)
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)


if __name__ == "__main__":
    main()