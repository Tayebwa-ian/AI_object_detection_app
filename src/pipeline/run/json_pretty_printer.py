"""print helper function to print in a json file with good formating"""
import os
import json
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, path: str):
    """Save dict to JSON with pretty formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
