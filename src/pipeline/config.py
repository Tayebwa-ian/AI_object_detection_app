"""
Configuration constants for the model pipeline.
Adjust paths and constants here.
"""
import torch
from pathlib import Path

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint paths (you can change these to where you store them)
BASE_DIR = Path("./src/pipeline")
SAM_CHECKPOINT = BASE_DIR / "sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"

# ResNet backbone type
RESNET_TYPE = "resnet50"  # torchvision name for resnet50
IMAGE_SIZE = (224, 224)   # default transform size for ResNet features

# CLIP model name (will attempt to load via the 'clip' package)
CLIP_MODEL = "ViT-B/32"
