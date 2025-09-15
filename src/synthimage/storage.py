#!/usr/bin/python3
"""
Storage manager that saves images under label folders with UUID-based unique filenames.
"""

import os
from pathlib import Path
from typing import Optional
from PIL import Image
import uuid
from .config import DEFAULT_LABELS_DIR, DEFAULT_OUTPUT_FORMAT


class StorageManager:
    """
    Creates label subfolders under a root directory and saves images with UUID-based unique filenames.

    Parameters
    ----------
    root_dir : str
        Base directory for storing images.
    output_format : str
        Output format for saved images (e.g., "PNG" or "JPEG").
    """

    def __init__(self, root_dir: str = DEFAULT_LABELS_DIR, output_format: str = DEFAULT_OUTPUT_FORMAT):
        """
        Initialize the StorageManager with a root directory and output format.

        Args:
            root_dir (str): Base directory for storing images. Defaults to DEFAULT_LABELS_DIR.
            output_format (str): Output format for saved images. Defaults to DEFAULT_OUTPUT_FORMAT.
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format

    def save_image(self, img: Image.Image, label: str, filename: Optional[str] = None) -> Path:
        """
        Save an image to <root_dir>/<label>/<filename>.

        Filenames are always generated using a UUID for uniqueness, prefixed with the label.
        If a custom filename is provided, it is used as a prefix but does not affect uniqueness.
        The file extension is determined by output_format.

        Args:
            img (Image.Image): The PIL Image object to save.
            label (str): The label subdirectory to store the image in.
            filename (Optional[str]): Optional custom filename prefix. Ignored for uniqueness.

        Returns:
            Path: The full path to the saved file.
        """
        # Ensure the label directory exists
        label_dir = self.root_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename using UUID
        unique_id = str(uuid.uuid4())  # Generates a random UUID (e.g., '123e4567-e89b-12d3-a456-426614174000')
        if filename:
            # Use provided filename as a prefix, but ensure uniqueness with UUID
            filename = f"{label}_{unique_id}.{self.output_format.lower()}"
        else:
            # Default to label and UUID
            filename = f"{label}_{unique_id}.{self.output_format.lower()}"

        # Save the image to the final path
        out_path = label_dir / filename
        img.save(out_path, format=self.output_format)
        return out_path
