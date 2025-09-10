"""
CSV Logger Module

Handles logging test results to CSV files.
"""

import csv
import logging
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class CSVLogger:
    """Logs test results to CSV."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_path = output_dir / f"test_results_{int(time.time())}.csv"
        
        # Create CSV with headers
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_path', 'object_type', 'truth', 'predicted', 'confidence', 
                'time_ms', 'segments'
            ])
        
        logger.info(f"CSV logging to: {self.csv_path}")
    
    def log_result(self, image_path: str, object_type: str, truth: int, 
                   predicted: int, confidence: float, time_ms: float, segments: int):
        """Log a single result to CSV."""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_path, object_type, truth, predicted, confidence, time_ms, segments])
