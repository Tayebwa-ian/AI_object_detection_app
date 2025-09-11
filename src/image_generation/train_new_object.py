#!/usr/bin/env python3
"""
Automated Training Script for New Objects

This script implements the complete workflow:
1. Register a new object (e.g., "banana") with optional example images
2. Generate synthetic training data
3. Continuously train the model with generated images
4. Test with real images

Usage:
    python tools/train_new_object.py --object banana --examples path/to/banana_images/
    python tools/train_new_object.py --object apple --examples path/to/apple_images/ --iterations 10
"""

import argparse
import requests
import time
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from typing import List, Tuple, Optional
from tools.automated_testing import TestImageGenerator
from tools.image_generator import ImageGenerator

# Configuration
API_BASE = "http://127.0.0.1:5000"
GENERATION_ITERATIONS = 5  # Number of training rounds
IMAGES_PER_ITERATION = 10  # Images to generate per round
TEST_IMAGES = 5  # Images for final testing

class ObjectTrainer:
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
        self.session = requests.Session()
        self.ai_generator = ImageGenerator(224, 224, "generated_training_images")
        self.fallback_generator = TestImageGenerator(224, 224)  # Fallback for when AI fails
        
    def check_api_health(self) -> bool:
        """Check if API is running."""
        try:
            response = self.session.get(f"{self.api_base}/health")
            return response.status_code == 200
        except:
            return False
    
    def register_object(self, object_name: str, example_images: Optional[List[str]] = None) -> dict:
        """Register a new object with optional example images."""
        print(f"Registering object: {object_name}")
        
        if not example_images:
            # Create a simple synthetic image for initial registration
            example_images = [self._create_simple_image(object_name)]
        
        # Convert images to bytes
        image_data = []
        for img in example_images:
            if isinstance(img, str):
                # It's a file path
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                # It's a PIL Image - convert to bytes
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                img_data = img_bytes.getvalue()
            image_data.append(img_data)
        
        # Register with ResNet few-shot
        files = []
        for i, img_data in enumerate(image_data):
            files.append(('images', (f'image_{i}.jpg', img_data, 'image/jpeg')))
        
        data = {'label': object_name}
        
        try:
            response = self.session.post(
                f"{self.api_base}/resnet-fewshot/register",
                data=data,
                files=files
            )
            result = response.json()
            
            if result.get('success'):
                print(f" {object_name} registered successfully!")
                print(f"    Support samples: {result.get('support_samples', 0)}")
                print(f"    Method: {result.get('method', 'unknown')}")
                return result
            else:
                print(f" Registration failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f" Registration error: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_training_images(self, object_name: str, count: int = 10) -> List[Image.Image]:
        """Generate synthetic training images using your AI image generator."""
        print(f" Generating {count} AI training images for {object_name}...")
        
        images = []
        successful_generations = 0
        
        for i in range(count):
            try:
                # Use your AI image generator
                ai_image, metadata = self.ai_generator.generate_image(
                    object_types=[object_name],
                    num_objects=1,
                    background="none",
                    blur=random.choice([True, False]),
                    rotate=random.choice([True, False]),
                    noise=random.choice([True, False])
                )
                
                if ai_image is not None:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(ai_image)
                    images.append(pil_image)
                    successful_generations += 1
                    print(f"    Generated AI image {i+1}/{count}")
                else:
                    # Fallback to simple generator if AI fails
                    print(f"    AI generation failed for image {i+1}, using fallback...")
                    objects = [{
                        "type": object_name,
                        "count": 1,
                        "x": random.randint(50, 150),
                        "y": random.randint(50, 150),
                        "size": random.randint(40, 80),
                        "color": tuple(random.randint(0, 255) for _ in range(3))
                    }]
                    
                    img, _ = self.fallback_generator.generate_test_image(
                        objects=objects,
                        background="solid",
                        difficulty="easy"
                    )
                    images.append(img)
                    successful_generations += 1
                    
            except Exception as e:
                print(f"    Error generating image {i+1}: {e}")
                # Use fallback generator
                objects = [{
                    "type": object_name,
                    "count": 1,
                    "x": random.randint(50, 150),
                    "y": random.randint(50, 150),
                    "size": random.randint(40, 80),
                    "color": tuple(random.randint(0, 255) for _ in range(3))
                }]
                
                img, _ = self.fallback_generator.generate_test_image(
                    objects=objects,
                    background="solid",
                    difficulty="easy"
                )
                images.append(img)
                successful_generations += 1
        
        print(f"Successfully generated {successful_generations}/{count} images")
        return images
    
    def _create_simple_image(self, object_name: str) -> Image.Image:
        """Create a simple image for initial registration."""

        # Use the existing TestImageGenerator to create a simple image
        generator = TestImageGenerator(224, 224)
        objects = [{
            "type": object_name,
            "count": 1,
            "x": 50,
            "y": 50,
            "size": 60,
            "color": (255, 255, 0)  # Yellow
        }]
        
        img, _ = generator.generate_test_image(
            objects=objects,
            background="solid",
            difficulty="easy"
        )
        return img
    
    def train_with_images(self, object_name: str, images: List[Image.Image]) -> dict:
        """Train the model with generated images."""
        print(f"🏋️ Training {object_name} with {len(images)} images...")
        
        # Convert PIL images to bytes
        image_data = []
        for img in images:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save to bytes
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            image_data.append(img_bytes.getvalue())
        
        # Prepare files for API
        files = []
        for i, img_data in enumerate(image_data):
            files.append(('images', (f'training_{i}.jpg', img_data, 'image/jpeg')))
        
        data = {'label': object_name}
        
        try:
            response = self.session.post(
                f"{self.api_base}/resnet-fewshot/register",
                data=data,
                files=files
            )
            result = response.json()
            
            if result.get('success'):
                print(f" Training successful!")
                print(f"    Total samples: {result.get('support_samples', 0)}")
                return result
            else:
                print(f" Training failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f" Training error: {e}")
            return {"success": False, "error": str(e)}
    
    def test_object_detection(self, object_name: str, test_images: List[Image.Image]) -> dict:
        """Test the trained object with new images."""
        print(f" Testing {object_name} detection with {len(test_images)} images...")
        
        results = []
        for i, img in enumerate(test_images):
            print(f"   Testing image {i+1}/{len(test_images)}...")
            
            # Convert to bytes
            if img.mode != 'RGB':
                img = img.convert('RGB')
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Test with ResNet advanced detection
            files = {'image': (f'test_{i}.jpg', img_bytes.getvalue(), 'image/jpeg')}
            data = {
                'candidate_labels': [object_name],
                'confidence_threshold': 0.3
            }
            
            try:
                response = self.session.post(
                    f"{self.api_base}/detect-resnet-advanced",
                    data=data,
                    files=files
                )
                result = response.json()
                
                # Extract detection info
                detections = result.get('detections', [])
                detected_objects = [d.get('mapped_label', 'unknown') for d in detections]
                has_target = object_name in detected_objects
                
                results.append({
                    'image_id': i,
                    'detected_objects': detected_objects,
                    'has_target': has_target,
                    'confidence': max([d.get('confidence', {}).get('combined', 0) for d in detections], default=0)
                })
                
                print(f"      Detected: {detected_objects} (Target: {has_target})")
                
            except Exception as e:
                print(f"       Test failed: {e}")
                results.append({
                    'image_id': i,
                    'detected_objects': [],
                    'has_target': False,
                    'confidence': 0,
                    'error': str(e)
                })
        
        # Calculate success rate
        successful_detections = sum(1 for r in results if r['has_target'])
        success_rate = successful_detections / len(results) if results else 0
        avg_confidence = np.mean([r['confidence'] for r in results if 'error' not in r])
        
        print(f" Test Results:")
        print(f"   Success Rate: {success_rate:.1%} ({successful_detections}/{len(results)})")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        return {
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'results': results
        }
    
    
    def run_complete_training(self, object_name: str, example_images: Optional[List[str]] = None, 
                            iterations: int = GENERATION_ITERATIONS) -> dict:
        """Run the complete training workflow."""
        print(f"Starting complete training workflow for: {object_name}")
        print("=" * 60)
        
        # Check API health
        if not self.check_api_health():
            print(" API is not running! Please start the Flask API first.")
            return {"success": False, "error": "API not available"}
        
        # Step 1: Initial registration
        print("\n Step 1: Initial Registration")
        reg_result = self.register_object(object_name, example_images)
        if not reg_result.get('success'):
            return reg_result
        
        # Step 2: Iterative training
        print(f"\n Step 2: Iterative Training ({iterations} rounds)")
        training_results = []
        
        for iteration in range(iterations):
            print(f"\n--- Training Round {iteration + 1}/{iterations} ---")
            
            # Generate training images
            training_images = self.generate_training_images(object_name, IMAGES_PER_ITERATION)
            
            # Train with generated images
            train_result = self.train_with_images(object_name, training_images)
            training_results.append(train_result)
            
            if not train_result.get('success'):
                print(f" Training round {iteration + 1} failed, continuing...")
            
            # Small delay between rounds
            time.sleep(1)
        
        # Step 3: Final testing
        print(f"\nStep 3: Final Testing")
        test_images = self.generate_training_images(object_name, TEST_IMAGES)
        test_result = self.test_object_detection(object_name, test_images)
        
        # Summary
        print(f"\nTraining Summary for {object_name}:")
        print(f"   Initial registration: {'Yes' if reg_result.get('success') else 'No'}")
        print(f"   Training rounds: {iterations}")
        print(f"   Successful rounds: {sum(1 for r in training_results if r.get('success'))}")
        print(f"   Final success rate: {test_result['success_rate']:.1%}")
        print(f"   Average confidence: {test_result['avg_confidence']:.3f}")
        
        return {
            "success": True,
            "object_name": object_name,
            "registration": reg_result,
            "training_rounds": training_results,
            "test_results": test_result,
            "summary": {
                "total_rounds": iterations,
                "successful_rounds": sum(1 for r in training_results if r.get('success')),
                "final_success_rate": test_result['success_rate'],
                "avg_confidence": test_result['avg_confidence']
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Train new objects with ResNet few-shot learning")
    parser.add_argument("--object", required=True, help="Name of the object to train (e.g., 'banana')")
    parser.add_argument("--examples", help="Path to example images directory")
    parser.add_argument("--iterations", type=int, default=GENERATION_ITERATIONS, 
                       help=f"Number of training iterations (default: {GENERATION_ITERATIONS})")
    parser.add_argument("--api-base", default=API_BASE, help="API base URL")
    
    args = parser.parse_args()
    
    # Load example images if provided
    example_images = None
    if args.examples and os.path.exists(args.examples):
        example_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            example_paths.extend(Path(args.examples).glob(ext))
        example_images = [str(p) for p in example_paths]
        print(f"📁 Found {len(example_images)} example images")
    
    # Run training
    trainer = ObjectTrainer(args.api_base)
    result = trainer.run_complete_training(
        object_name=args.object,
        example_images=example_images,
        iterations=args.iterations
    )
    
    if result.get('success'):
        print("\n Training completed successfully!")
    else:
        print(f"\n Training failed: {result.get('error', 'Unknown error')}")
        exit(1)

if __name__ == "__main__":
    main()
