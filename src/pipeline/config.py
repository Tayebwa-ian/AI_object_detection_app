#!/usr/bin/python3"
"""
Configuration constants for the model pipeline.
Adjust paths and constants here.
"""
import torch
from pathlib import Path

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint paths (you can change these to where you store them)
BASE_DIR = Path("./src/pipeline/store")
SAM_CHECKPOINT = BASE_DIR / "sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "data"
DEFAULT_TEST_ROOT = BASE_DIR / "test"
DEFAULT_CLASSIFIER_PATH = BASE_DIR / "classifier/fewshot_clf.joblib"
LOGISTIC_CLASSIFIER_PATH = BASE_DIR / "classifier/logistic_clf.joblib"
LINEAR_SVC_CLASSIFIER_PATH = BASE_DIR / "classifier/linear_clf.joblib"

# ResNet backbone type
RESNET_TYPE = "resnet50"  # torchvision name for resnet50
IMAGE_SIZE = (224, 224)   # default transform size for ResNet features

# CLIP model name (will attempt to load via the 'clip' package)
CLIP_MODEL = "ViT-B/32"
