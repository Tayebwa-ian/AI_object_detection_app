"""
API Tester Module

Handles testing images against the API and submitting corrections.
"""

import logging
import time
from typing import Any, Dict

import requests

# Configure logging
logger = logging.getLogger(__name__)


class APITester:
    """Tests images against the API and submits corrections."""
    
    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
    
    def test_image(self, image_path: str, object_type: str, expected_count: int) -> Dict[str, Any]:
        """POST image to /api/count and return results."""
        try:
            url = f"{self.api_url}/api/count"
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'object_type': object_type}
                
                start_time = time.time()
                response = requests.post(url, files=files, data=data, timeout=self.timeout)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_count = result.get('predicted_count', 0)
                    confidence = result.get('confidence', 0.0)
                    segments = result.get('segments', [])
                    result_id = result.get('result_id') or result.get('output_id')
                    
                    return {
                        'success': True,
                        'result_id': result_id,
                        'predicted_count': predicted_count,
                        'expected_count': expected_count,
                        'confidence': confidence,
                        'segments': len(segments),
                        'response_time_ms': response_time,
                        'raw_response': result
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API error: {response.status_code}",
                        'response_time_ms': response_time,
                        'predicted_count': 0,
                        'expected_count': expected_count,
                        'confidence': 0.0,
                        'segments': 0
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': 0,
                'predicted_count': 0,
                'expected_count': expected_count,
                'confidence': 0.0,
                'segments': 0
            }
    
    def submit_correction(self, result_id: int, corrected_count: int) -> bool:
        """PUT correction to /api/correct/<result_id>."""
        try:
            url = f"{self.api_url}/api/correct/{result_id}"
            data = {'corrected_count': corrected_count}
            
            response = requests.put(url, json=data, timeout=self.timeout)
            
            if response.status_code == 200:
                logger.info(f"Submitted correction: result_id={result_id}, count={corrected_count}")
                return True
            else:
                logger.error(f"Correction failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Correction error: {e}")
            return False
