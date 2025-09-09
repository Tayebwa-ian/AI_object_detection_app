#!/usr/bin/env python3
"""
AI Object Counter - Image Generation and API Testing Script (Refactored)

Generates images using AI endpoint, posts to API, and logs results to CSV.
Follows exact requirements for Task 2.

Usage:
    python tools/generate_and_post_refactored.py --num-images 5 --objects car,person --min-objects 1 --max-objects 3
    python tools/generate_and_post_refactored.py --num-images 10 --objects car --background red --blur --rotate --noise
    python tools/generate_and_post_refactored.py --num-images 3 --size 1024x768 --api http://localhost:3000
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List

from image_generation_scripts import ImageGenerator, APITester, CSVLogger

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
