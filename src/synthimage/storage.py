#!/usr/bin/python3"
"""
Storage manager that saves images under label folders with systematic unique filenames.
"""

import os
from pathlib import Path
from typing import Optional
from PIL import Image
from .config import DEFAULT_LABELS_DIR, DEFAULT_OUTPUT_FORMAT


class StorageManager:
    """
    Creates label subfolders under a root dir and saves images into them.

    Parameters
    ----------
    root_dir: str
        Base directory for storing images.
    output_format: str
        Output format for saved images (e.g. "PNG" or "JPEG").
    """

    def __init__(self, root_dir: str = DEFAULT_LABELS_DIR, output_format: str = DEFAULT_OUTPUT_FORMAT):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format

    def save_image(self, img: Image.Image, label: str, filename: Optional[str] = None) -> Path:
        """
        Save an image to <root_dir>/<label>/<filename>.

        If filename is not provided, a unique name is generated automatically.
        If filename is provided but already exists, a numeric suffix is appended.

        Returns
        -------
        Path
            The full path to the saved file.
        """
        label_dir = self.root_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            # Generate unique filename using an incrementing counter
            existing = list(label_dir.glob(f"*.{self.output_format.lower()}"))
            if existing:
                # extract existing numeric indices
                indices = []
                for f in existing:
                    stem = f.stem
                    parts = stem.split("_")
                    if parts and parts[-1].isdigit():
                        indices.append(int(parts[-1]))
                idx = max(indices) + 1 if indices else 1
            else:
                idx = 1
            filename = f"{label}_{idx:04d}.{self.output_format.lower()}"
        else:
            # Ensure proper extension
            if not filename.lower().endswith(f".{self.output_format.lower()}"):
                filename = f"{filename}.{self.output_format.lower()}"

            # Ensure uniqueness by appending a suffix if needed
            out_path = label_dir / filename
            counter = 1
            while out_path.exists():
                stem = Path(filename).stem
                ext = Path(filename).suffix
                filename = f"{stem}_{counter}{ext}"
                out_path = label_dir / filename
                counter += 1

        out_path = label_dir / filename
        img.save(out_path, format=self.output_format)
        return out_path
