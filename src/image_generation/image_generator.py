"""
Image Generator Module

Handles AI image generation using external endpoints with optional transformations.
"""

import base64
import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter

# Configure logging
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
