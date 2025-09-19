#!/usr/bin/python3"
"""
Configuration constants for the api.
Adjust paths and constants here.
"""
from pathlib import Path

# Checkpoint paths (you can change these to where you store them)
BASE_STORAGE = Path("./Databases")
IMAGE_STORAGE = BASE_STORAGE / "uploaded_images"
