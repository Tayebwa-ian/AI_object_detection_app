#!/usr/bin/python3"
"""
synthimage.api_client
---------------------

Image generation client that calls an external AI image generation endpoint.

This implementation borrows the request structure and response handling you
used previously (the `/v1/images/generations` style returning base64 JSON
under `data[0].b64_json`) but improves structure, logging, typing, and uses
the package utilities for consistent sizing and format.

Only this module/file was changed to align the package's network calls with
your working script's format/behavior.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from .config import DEFAULT_IMAGE_SIZE
from .utils import pil_to_base64_str, base64_str_to_pil, ensure_rgb, center_crop_resize

# Module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ImageGenClient:
    """
    Client to generate images via a remote AI image endpoint.

    This client:
      - uses a requests.Session for connection reuse,
      - supports payload fields similar to the script you provided
        (e.g. "model", "sampling_steps", "cfg_scale", "prompt", etc.),
      - accepts either a textual label (prompt) or an example PIL image (or both),
      - rounds requested width/height up to multiples of 64 (many diffusion
        backends require sizes divisible by 64),
      - decodes responses that return base64 content under 'data' -> 'b64_json'.

    Parameters
    ----------
    endpoint: str
        Full base URL of the AI endpoint (for example:
        "https://llm-web.aieng.fim.uni-passau.de/v1/images/generations").
    token: str
        Bearer token to put into the Authorization header.
    default_size: tuple[int, int]
        Desired output size (width, height). The width/height will be rounded
        up to the nearest multiple of 64 when sent to the server.
    session: Optional[requests.Session]
        Optional custom session. If omitted, a new session is created and the
        Authorization header is added automatically.
    timeout: float
        Default request timeout in seconds.
    verify_ssl: bool
        Whether to verify server TLS certificate.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        default_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        session: Optional[requests.Session] = None,
        timeout: float = 60.0,
        verify_ssl: bool = True,
    ):
        self.endpoint = endpoint.rstrip("/")  # ensure no trailing slash
        self.token = token
        self.default_size = default_size
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Use provided session or create one and set Authorization header
        self.session = session or requests.Session()
        # Set the header for every request (Bearer token auth)
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    # ---------- Public API ----------
    def generate(
        self,
        label: Optional[str] = None,
        example_image: Optional[Image.Image] = None,
        n: int = 1,
        size: Optional[Tuple[int, int]] = None,
        model: str = "flux.1-schnell-gguf",
        sampling_steps: int = 20,
        sample_method: str = "euler",
        cfg_scale: float = 1.0,
        guidance: float = 3.5,
        strength: float = 0.75,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Image.Image]:
        """
        Generate images from the remote API.

        Parameters
        ----------
        label:
            Text prompt to send to the model (optional).
        example_image:
            PIL Image to pass to the model as an initial image (optional).
        n:
            Number of images to request. If the server returns fewer images,
            the returned list may be padded by repeating the last received image.
        size:
            (width, height). If omitted, `self.default_size` is used. The values
            sent to the server are rounded up to the nearest multiple of 64.
        model, sampling_steps, sample_method, cfg_scale, guidance, strength:
            Generation parameters forwarded to the endpoint (defaults chosen
            to match your original script).
        extra_payload:
            A dictionary of additional fields to include in the JSON body.

        Returns
        -------
        List[PIL.Image]
            A list of PIL Images (RGB), each resized/cropped to the requested
            size (center-crop + LANCZOS).
        """
        size = size or self.default_size
        rounded_w, rounded_h = _round_up_to_multiple_of_64(size[0]), _round_up_to_multiple_of_64(size[1])
        requested_size_str = f"{rounded_w}x{rounded_h}"

        # Build payload resembling your previous working script
        payload: Dict[str, Any] = {
            "n": n,
            "size": requested_size_str,
            "seed": None,
            "sample_method": sample_method,
            "cfg_scale": cfg_scale,
            "guidance": guidance,
            "sampling_steps": sampling_steps,
            "negative_prompt": "",
            "strength": strength,
            "schedule_method": "discrete",
            "model": model,
        }

        # Add prompt if provided
        if label:
            payload["prompt"] = label

        # If example image is provided, include it as base64 in the JSON body.
        # (If your endpoint prefers multipart uploads, you can switch to that
        # mode separately — this implementation follows the JSON + base64 approach.)
        if example_image is not None:
            example_image = ensure_rgb(example_image)
            payload["image"] = pil_to_base64_str(example_image, fmt="PNG")

        if extra_payload:
            payload.update(extra_payload)

        url = f"{self.endpoint}/v1/images/generations" if not self.endpoint.endswith("/v1/images/generations") and "/v1/images/generations" not in self.endpoint else self.endpoint

        logger.info("Requesting image generation: prompt=%s size=%s n=%d model=%s", 
                    (label[:60] + "..." if label else "<none>"), requested_size_str, n, model)

        try:
            resp = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.exceptions.ConnectionError as exc:
            logger.error("AI endpoint connection failed: %s", exc)
            logger.debug("Attempted URL: %s", url)
            raise
        except requests.exceptions.Timeout as exc:
            logger.error("AI endpoint timeout after %s seconds: %s", self.timeout, exc)
            raise

        # Non-2xx handling
        if resp.status_code != 200:
            # try to include server message in logs
            text = resp.text[:1000] if resp.text else "<no-body>"
            logger.error("AI endpoint error: %d - %s", resp.status_code, text)
            resp.raise_for_status()

        # Try to parse the common "data" -> [{'b64_json': ...}, ...] format
        images: List[Image.Image] = []
        content_type = resp.headers.get("Content-Type", "")

        try:
            if "application/json" in content_type or resp.text.strip().startswith("{"):
                body = resp.json()
                # First, support the pattern you used: result['data'][0]['b64_json']
                if isinstance(body, dict) and "data" in body and isinstance(body["data"], list):
                    for item in body["data"]:
                        if isinstance(item, dict) and "b64_json" in item:
                            b64 = item["b64_json"]
                            pil = _pil_from_base64_or_dataurl(b64)
                            images.append(pil)
                        else:
                            # try other keys that may hold image content
                            for k in ("b64", "image", "image_base64"):
                                if isinstance(item, dict) and k in item:
                                    images.append(_pil_from_base64_or_dataurl(item[k]))
                                    break
                # Fallback: some servers return `images` list of base64 strings
                elif isinstance(body, dict) and "images" in body and isinstance(body["images"], list):
                    for entry in body["images"]:
                        if isinstance(entry, str):
                            images.append(_pil_from_base64_or_dataurl(entry))
                else:
                    # Attempt to find any base64-like strings in top-level dict values
                    for v in (body.values() if isinstance(body, dict) else []):
                        if isinstance(v, str) and _looks_like_base64(v):
                            images.append(_pil_from_base64_or_dataurl(v))
            elif "image/" in content_type:
                # Direct image bytes -> single image
                pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                images.append(pil)
            else:
                # Unknown content-type but try JSON anyway
                body = resp.json()
                logger.debug("Parsed JSON fallback body keys: %s", list(body.keys()) if isinstance(body, dict) else "<non-dict>")
        except Exception as exc:
            logger.exception("Failed to parse AI response body: %s", exc)
            raise

        # Ensure we have at least one image
        if not images:
            logger.error("AI endpoint returned no images in response.")
            raise RuntimeError("No images returned by AI endpoint")

        # Post-process images: ensure RGB and consistent final size (center crop + resize)
        final_size = size or self.default_size
        processed: List[Image.Image] = []
        for img in images[:n]:
            img = ensure_rgb(img)
            img = center_crop_resize(img, final_size)
            processed.append(img)

        # If the server returned fewer images than requested, repeat last to pad to n
        while len(processed) < n:
            processed.append(processed[-1].copy())

        return processed


# ---------- Helper functions ----------
def _round_up_to_multiple_of_64(x: int) -> int:
    """Round integer x up to nearest multiple of 64 (>= 64)."""
    if x <= 0:
        return 64
    return ((x + 63) // 64) * 64


def _pil_from_base64_or_dataurl(b64_or_dataurl: str) -> Image.Image:
    """
    Convert either a raw base64 string or a data:...;base64,... data URL into a PIL Image.

    The returned image is converted to RGB.
    """
    # If data URL like "data:image/png;base64,...."
    if b64_or_dataurl.startswith("data:"):
        _, payload = b64_or_dataurl.split(",", 1)
        b64_data = payload
    else:
        b64_data = b64_or_dataurl

    try:
        img = base64_str_to_pil(b64_data)
    except Exception:
        # fallback: try to decode raw base64 bytes
        raw = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.convert("RGB")


def _looks_like_base64(s: str) -> bool:
    """Quick heuristic if a string looks like base64 data (not exhaustive)."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) < 20:
        return False
    # Many base64 payloads contain only A-Za-z0-9+/= characters
    try:
        # Try to decode; if it fails, not base64
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        # Could be a data URL (handle above) or not base64
        return s.startswith("data:")

