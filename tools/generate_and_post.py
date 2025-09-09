#!/usr/bin/env python3
"""
AI Object Counter - Image Generation and API Testing Script

Generates images using AI endpoint, posts to API, and logs results to CSV.
Follows exact requirements for Task 2.

Usage:
    python tools/generate_and_post.py --num-images 5 --objects car,person --min-objects 1 --max-objects 3
    python tools/generate_and_post.py --num-images 10 --objects car --background red --blur --rotate --noise
    python tools/generate_and_post.py --num-images 3 --size 1024x768 --api http://localhost:3000
"""

import argparse
import base64
import csv
import io
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tools/generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AI endpoint configuration
AI_ENDPOINT = "llm-web.aieng.fim.uni-passau.de"
AI_API_KEY = "gpustack_adf7d482bd8a814b_a1bfc829fc58b64de0d65cdd91473815"


class ImageGenerator:
    """Generates images using AI endpoint with optional PIL compositing."""
    
    def __init__(self, width: int, height: int, output_dir: str):
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_ai_image(self, prompt: str) -> Optional[np.ndarray]:
        """Generate image using AI endpoint."""
        try:
            url = f"https://{AI_ENDPOINT}/v1/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AI_API_KEY}"
            }
            
            # Convert size to the format expected by the API (must be multiples of 64)
            # Round to nearest multiple of 64
            width_64 = ((self.width + 63) // 64) * 64
            height_64 = ((self.height + 63) // 64) * 64
            size = f"{width_64}x{height_64}"
            
            payload = {
                "n": 1,
                "size": size,
                "seed": None,
                "sample_method": "euler",
                "cfg_scale": 1,
                "guidance": 3.5,
                "sampling_steps": 20,
                "negative_prompt": "",
                "strength": 0.75,
                "schedule_method": "discrete",
                "model": "flux.1-schnell-gguf",
                "prompt": prompt
            }
            
            logger.info(f"Generating AI image: {prompt[:50]}...")
            response = requests.post(url, json=payload, headers=headers, timeout=60, verify=True)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle the response format from the API
                if 'data' in result and result['data'] and len(result['data']) > 0:
                    image_data = result['data'][0]['b64_json']
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    return np.array(image.resize((self.width, self.height), Image.Resampling.LANCZOS))
                else:
                    logger.error(f"Unexpected AI response format: {result}")
                    return None
            
            else:
                logger.error(f"AI endpoint error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"AI endpoint connection failed: {e}")
            logger.error("This might be due to:")
            logger.error("1. Network connectivity issues")
            logger.error("2. VPN not connected (if required)")
            logger.error("3. AI endpoint is down")
            logger.error("4. Incorrect endpoint URL")
            logger.error(f"Trying to reach: https://{AI_ENDPOINT}/v1/images/generations")
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"AI endpoint timeout: {e}")
            logger.error("The AI endpoint took too long to respond")
            return None
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return None
    
    def apply_transformations(self, image: np.ndarray, blur: bool, rotate: bool, noise: bool) -> np.ndarray:
        """Apply transformations using PIL."""
        pil_image = Image.fromarray(image)
        
        if blur:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        if rotate:
            angle = random.uniform(-15, 15)
            pil_image = pil_image.rotate(angle, expand=False)
        
        if noise:
            arr = np.array(pil_image).astype(np.float32)
            noise_array = np.random.normal(0, 15, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise_array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(arr)
        
        return np.array(pil_image)
    
    def generate_image(self, object_types: List[str], num_objects: int, background: str, 
                      blur: bool, rotate: bool, noise: bool) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Generate a single image with specified parameters."""
        
        # Create prompt based on object types and count
        if num_objects == 1:
            count_desc = "one"
        elif num_objects == 2:
            count_desc = "two"
        else:
            count_desc = f"{num_objects}"
        
        if len(object_types) == 1:
            obj_type = object_types[0]
            if num_objects == 1:
                prompt = f"A single {obj_type}"
            else:
                prompt = f"{count_desc} {obj_type}s"
        else:
            obj_list = ", ".join(object_types[:-1]) + f" and {object_types[-1]}"
            prompt = f"A scene with {obj_list}"
        
        # Add background context
        if background and background != "none":
            if os.path.exists(background):
                prompt += f" with a custom background"
            else:
                prompt += f" on a {background} background"
        
        # Add setting variety
        settings = ["in a park", "on a street", "in a parking lot", "in a garden", "on a sidewalk"]
        prompt += f" {random.choice(settings)}"
        
        # Generate AI image
        ai_image = self.generate_ai_image(prompt)
        
        if ai_image is None:
            return None, {}
        
        # Apply transformations
        final_image = self.apply_transformations(ai_image, blur, rotate, noise)
        
        # Create metadata
        metadata = {
            'prompt': prompt,
            'object_types': object_types,
            'expected_objects': num_objects,
            'background': background,
            'transformations': {
                'blur': blur,
                'rotate': rotate,
                'noise': noise
            },
            'image_size': (self.width, self.height)
        }
        
        return final_image, metadata
    
    def save_image(self, image: np.ndarray, metadata: Dict[str, Any], filename: str) -> str:
        """Save image and metadata."""
        image_path = self.output_dir / f"{filename}.jpg"
        metadata_path = self.output_dir / f"{filename}_metadata.json"
        
        # Save image
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(image_path)


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate images and test API')
    
    # Required arguments
    parser.add_argument('--num-images', type=int, required=True, help='Number of images to generate')
    parser.add_argument('--objects', type=str, required=True, help='Comma-separated object types (e.g., car,person)')
    
    # Optional arguments
    parser.add_argument('--min-objects', type=int, default=1, help='Minimum objects per image')
    parser.add_argument('--max-objects', type=int, default=5, help='Maximum objects per image')
    parser.add_argument('--background', type=str, default='none', help='Background: path to image or color name')
    parser.add_argument('--blur', action='store_true', help='Apply blur transformation')
    parser.add_argument('--rotate', action='store_true', help='Apply rotation transformation')
    parser.add_argument('--noise', action='store_true', help='Apply noise transformation')
    parser.add_argument('--size', type=str, default='800x600', help='Image size as WxH (e.g., 1024x768)')
    parser.add_argument('--api', type=str, default='http://localhost:5000', help='API base URL')
    parser.add_argument('--output-dir', type=str, default='tools/generated_images', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Parse size
    try:
        width, height = map(int, args.size.split('x'))
    except ValueError:
        logger.error(f"Invalid size format: {args.size}. Use WxH (e.g., 800x600)")
        return
    
    # Parse objects
    object_types = [obj.strip() for obj in args.objects.split(',') if obj.strip()]
    if not object_types:
        logger.error("No valid object types specified")
        return
    
    logger.info(f"Generating {args.num_images} images")
    logger.info(f"Object types: {object_types}")
    logger.info(f"Size: {width}x{height}")
    logger.info(f"API: {args.api}")
    
    # Initialize components
    generator = ImageGenerator(width, height, args.output_dir)
    tester = APITester(args.api)
    csv_logger = CSVLogger(Path(args.output_dir))
    
    # Generate and test images
    for i in range(args.num_images):
        logger.info(f"Generating image {i+1}/{args.num_images}")
        
        # Random number of objects
        num_objects = random.randint(args.min_objects, args.max_objects)
        
        # Generate image
        image, metadata = generator.generate_image(
            object_types, num_objects, args.background, 
            args.blur, args.rotate, args.noise
        )
        
        if image is None:
            logger.error(f"Failed to generate image {i+1}")
            continue
        
        # Save image
        filename = f"test_image_{i+1:03d}"
        image_path = generator.save_image(image, metadata, filename)
        logger.info(f"Saved: {image_path}")
        
        # Choose object type for testing (most common in metadata)
        test_object_type = random.choice(object_types)
        
        # Test against API
        logger.info(f"Testing image {i+1} with object_type={test_object_type}, expected={num_objects}")
        result = tester.test_image(image_path, test_object_type, num_objects)
        
        # Log to CSV
        csv_logger.log_result(
            image_path=image_path,
            object_type=test_object_type,
            truth=num_objects,
            predicted=result.get('predicted_count', 0),
            confidence=result.get('confidence', 0.0),
            time_ms=result.get('response_time_ms', 0),
            segments=result.get('segments', 0)
        )
        
        # Submit correction if we have a result_id
        if result.get('success') and result.get('result_id'):
            tester.submit_correction(result['result_id'], num_objects)
        
        # Log result
        if result.get('success'):
            logger.info(f"SUCCESS Image {i+1}: predicted={result['predicted_count']}, expected={num_objects}, confidence={result['confidence']:.3f}")
        else:
            logger.error(f"X Image {i+1}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"Completed! Results logged to: {csv_logger.csv_path}")


if __name__ == '__main__':
    main()
