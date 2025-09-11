#!/usr/bin/env python3
"""
Automated image generation and API testing script.

This script:
1. Generates configurable test images with ground truth
2. Posts them to the API endpoints
3. Compares results with ground truth
4. Computes comprehensive metrics
5. Exports results to Prometheus format
"""

import argparse
import json
import time
import random
import requests
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pathlib import Path


class TestImageGenerator:
    """Generate test images with configurable objects and backgrounds."""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        
    def generate_solid_background(self, color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
        """Generate a solid color background."""
        return Image.new('RGB', (self.width, self.height), color)
    
    def generate_texture_background(self, pattern: str = "noise") -> Image.Image:
        """Generate a textured background."""
        if pattern == "noise":
            noise = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            return Image.fromarray(noise)
        elif pattern == "gradient":
            gradient = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                gradient[y, :] = [int(255 * y / self.height)] * 3
            return Image.fromarray(gradient)
        else:
            return self.generate_solid_background()
    
    def draw_object(self, img: Image.Image, object_type: str, x: int, y: int, 
                   size: int, color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
        """Draw a simple object on the image."""
        draw = ImageDraw.Draw(img)
        
        if object_type == "person":
            # Draw a stick figure
            center_x, center_y = x + size//2, y + size//2
            # Head
            draw.ellipse([center_x-10, center_y-20, center_x+10, center_y], fill=color)
            # Body
            draw.line([center_x, center_y, center_x, center_y+30], fill=color, width=3)
            # Arms
            draw.line([center_x, center_y+10, center_x-15, center_y+20], fill=color, width=3)
            draw.line([center_x, center_y+10, center_x+15, center_y+20], fill=color, width=3)
            # Legs
            draw.line([center_x, center_y+30, center_x-10, center_y+50], fill=color, width=3)
            draw.line([center_x, center_y+30, center_x+10, center_y+50], fill=color, width=3)
            
        elif object_type == "car":
            # Draw a simple car
            draw.rectangle([x, y+size//3, x+size, y+size], fill=color)
            # Wheels
            wheel_y = y + size*2//3
            wheel_size = size//6
            draw.ellipse([x+5, wheel_y, x+5+wheel_size, wheel_y+wheel_size], fill=(0, 0, 0))
            draw.ellipse([x+size-wheel_size-5, wheel_y, x+size-5, wheel_y+wheel_size], fill=(0, 0, 0))
            
        elif object_type == "bicycle":
            # Draw a simple bicycle
            center_x, center_y = x + size//2, y + size//2
            # Wheels
            draw.ellipse([center_x-20, center_y-20, center_x+20, center_y+20], outline=color, width=3)
            draw.ellipse([center_x-15, center_y-15, center_x+15, center_y+15], outline=color, width=2)
            # Frame
            draw.line([center_x, center_y, center_x, center_y+30], fill=color, width=3)
            draw.line([center_x, center_y+15, center_x+20, center_y+25], fill=color, width=3)
            
        elif object_type == "mug":
            # Draw a simple mug
            draw.rectangle([x+size//4, y, x+size*3//4, y+size*2//3], fill=color)
            draw.rectangle([x+size*3//4, y+size//6, x+size*5//6, y+size//3], fill=color)
            
        return img
    
    def generate_test_image(self, objects: List[Dict[str, Any]], 
                          background: str = "solid", 
                          difficulty: str = "easy") -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate a test image with specified objects."""
        
        # Generate background
        if background == "solid":
            img = self.generate_solid_background()
        else:
            img = self.generate_texture_background(background)
        
        # Ground truth tracking
        ground_truth = {
            "objects": [],
            "total_count": 0,
            "image_size": (self.width, self.height),
            "background": background,
            "difficulty": difficulty
        }
        
        # Add objects
        for obj in objects:
            obj_type = obj["type"]
            count = obj.get("count", 1)
            
            for i in range(count):
                # Position based on difficulty
                if difficulty == "easy":
                    x = random.randint(50, self.width - 100)
                    y = random.randint(50, self.height - 100)
                    size = random.randint(40, 80)
                elif difficulty == "hard":
                    x = random.randint(10, self.width - 50)
                    y = random.randint(10, self.height - 50)
                    size = random.randint(20, 40)
                else:  # medium
                    x = random.randint(30, self.width - 80)
                    y = random.randint(30, self.height - 80)
                    size = random.randint(30, 60)
                
                # Apply transformations based on difficulty
                if difficulty == "hard":
                    # Add some randomness to make detection harder
                    if random.random() < 0.3:
                        size = max(10, size // 2)  # Make some objects very small
                
                img = self.draw_object(img, obj_type, x, y, size)
                ground_truth["objects"].append({
                    "type": obj_type,
                    "bbox": [x, y, x+size, y+size],
                    "center": [x+size//2, y+size//2]
                })
                ground_truth["total_count"] += 1
        
        return img, ground_truth


class APITester:
    """Test the detection API with generated images."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.results = []
    
    def test_standard_detection(self, image: Image.Image, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Test standard detection endpoint."""
        start_time = time.time()
        
        # Convert image to bytes
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Prepare request
        files = {'file': ('test.png', img_bytes, 'image/png')}
        data = {
            'candidate_labels': json.dumps([obj["type"] for obj in ground_truth["objects"]]),
            'confidence_threshold': '0.3'
        }
        
        try:
            response = requests.post(f"{self.base_url}/detect", files=files, data=data, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "duration": duration,
                    "status_code": response.status_code,
                    "detected_objects": result.get("summary", {}).get("total_objects", 0),
                    "detected_classes": result.get("summary", {}).get("classes_detected", []),
                    "avg_confidence": result.get("summary", {}).get("avg_combined_confidence", 0.0),
                    "stage_times": result.get("summary", {}).get("stage_times", {}),
                    "ground_truth_count": ground_truth["total_count"],
                    "mode": "standard"
                }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "status_code": response.status_code,
                    "error": response.text,
                    "ground_truth_count": ground_truth["total_count"],
                    "mode": "standard"
                }
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "ground_truth_count": ground_truth["total_count"],
                "mode": "standard"
            }
    
    def test_advanced_detection(self, image: Image.Image, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Test advanced detection endpoint (few-shot)."""
        start_time = time.time()
        
        # Convert image to bytes
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Prepare request
        files = {'file': ('test.png', img_bytes, 'image/png')}
        data = {
            'allowed_labels': json.dumps([obj["type"] for obj in ground_truth["objects"]]),
            'similarity_threshold': '0.5'
        }
        
        try:
            response = requests.post(f"{self.base_url}/detect-advanced", files=files, data=data, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "duration": duration,
                    "status_code": response.status_code,
                    "detected_objects": result.get("summary", {}).get("total_objects", 0),
                    "detected_classes": result.get("summary", {}).get("classes_detected", []),
                    "avg_confidence": result.get("summary", {}).get("avg_combined_confidence", 0.0),
                    "stage_times": result.get("summary", {}).get("stage_times", {}),
                    "ground_truth_count": ground_truth["total_count"],
                    "mode": "advanced"
                }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "status_code": response.status_code,
                    "error": response.text,
                    "ground_truth_count": ground_truth["total_count"],
                    "mode": "advanced"
                }
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "ground_truth_count": ground_truth["total_count"],
                "mode": "advanced"
            }
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive metrics from test results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("success", False)]
        
        # Basic metrics
        total_tests = len(results)
        successful_tests = len(successful_results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Accuracy metrics
        exact_matches = sum(1 for r in successful_results 
                          if r.get("detected_objects", 0) == r.get("ground_truth_count", 0))
        exact_match_accuracy = exact_matches / successful_tests if successful_tests > 0 else 0
        
        within_tolerance = sum(1 for r in successful_results 
                             if abs(r.get("detected_objects", 0) - r.get("ground_truth_count", 0)) <= 1)
        within_tolerance_accuracy = within_tolerance / successful_tests if successful_tests > 0 else 0
        
        # Timing metrics
        durations = [r.get("duration", 0) for r in successful_results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Confidence metrics
        confidences = [r.get("avg_confidence", 0) for r in successful_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Per-mode metrics
        mode_metrics = {}
        for mode in ["standard", "advanced"]:
            mode_results = [r for r in successful_results if r.get("mode") == mode]
            if mode_results:
                mode_metrics[mode] = {
                    "count": len(mode_results),
                    "exact_match_accuracy": sum(1 for r in mode_results 
                                              if r.get("detected_objects", 0) == r.get("ground_truth_count", 0)) / len(mode_results),
                    "avg_duration": sum(r.get("duration", 0) for r in mode_results) / len(mode_results),
                    "avg_confidence": sum(r.get("avg_confidence", 0) for r in mode_results) / len(mode_results)
                }
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "exact_match_accuracy": exact_match_accuracy,
            "within_tolerance_accuracy": within_tolerance_accuracy,
            "avg_duration_seconds": avg_duration,
            "min_duration_seconds": min_duration,
            "max_duration_seconds": max_duration,
            "avg_confidence": avg_confidence,
            "mode_metrics": mode_metrics
        }
    
    def export_to_prometheus(self, metrics: Dict[str, Any], output_file: str):
        """Export metrics to Prometheus format."""
        with open(output_file, 'w') as f:
            f.write("# HELP detection_tests_total Total number of detection tests\n")
            f.write("# TYPE detection_tests_total counter\n")
            f.write(f"detection_tests_total {metrics.get('total_tests', 0)}\n")
            
            f.write("# HELP detection_success_rate Detection success rate\n")
            f.write("# TYPE detection_success_rate gauge\n")
            f.write(f"detection_success_rate {metrics.get('success_rate', 0):.6f}\n")
            
            f.write("# HELP detection_exact_match_accuracy Exact match accuracy\n")
            f.write("# TYPE detection_exact_match_accuracy gauge\n")
            f.write(f"detection_exact_match_accuracy {metrics.get('exact_match_accuracy', 0):.6f}\n")
            
            f.write("# HELP detection_within_tolerance_accuracy Within tolerance accuracy\n")
            f.write("# TYPE detection_within_tolerance_accuracy gauge\n")
            f.write(f"detection_within_tolerance_accuracy {metrics.get('within_tolerance_accuracy', 0):.6f}\n")
            
            f.write("# HELP detection_avg_duration_seconds Average detection duration\n")
            f.write("# TYPE detection_avg_duration_seconds gauge\n")
            f.write(f"detection_avg_duration_seconds {metrics.get('avg_duration_seconds', 0):.6f}\n")
            
            f.write("# HELP detection_avg_confidence Average confidence score\n")
            f.write("# TYPE detection_avg_confidence gauge\n")
            f.write(f"detection_avg_confidence {metrics.get('avg_confidence', 0):.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Automated API testing with image generation")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000", help="API base URL")
    parser.add_argument("--num-tests", type=int, default=10, help="Number of test images to generate")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for results")
    parser.add_argument("--image-size", type=int, default=512, help="Test image size")
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"], 
                       help="Difficulty levels to test")
    parser.add_argument("--object-types", nargs="+", default=["person", "car", "bicycle", "mug"],
                       help="Object types to test")
    parser.add_argument("--export-prometheus", action="store_true", help="Export metrics to Prometheus format")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    generator = TestImageGenerator(args.image_size, args.image_size)
    tester = APITester(args.api_url)
    
    print(f"Starting automated testing with {args.num_tests} tests...")
    print(f"API URL: {args.api_url}")
    print(f"Output directory: {args.output_dir}")
    
    # Generate test cases
    test_cases = []
    for i in range(args.num_tests):
        difficulty = random.choice(args.difficulties)
        num_objects = random.randint(1, 3)
        
        objects = []
        for _ in range(num_objects):
            obj_type = random.choice(args.object_types)
            objects.append({
                "type": obj_type,
                "count": 1
            })
        
        # Generate image
        image, ground_truth = generator.generate_test_image(
            objects, 
            background=random.choice(["solid", "noise", "gradient"]),
            difficulty=difficulty
        )
        
        # Save test image
        image_path = os.path.join(args.output_dir, f"test_{i:03d}.png")
        image.save(image_path)
        
        test_cases.append({
            "image_path": image_path,
            "image": image,
            "ground_truth": ground_truth,
            "test_id": i
        })
    
    print(f"Generated {len(test_cases)} test cases")
    
    # Run tests
    all_results = []
    for i, test_case in enumerate(test_cases):
        print(f"Running test {i+1}/{len(test_cases)}...")
        
        # Test standard detection
        standard_result = tester.test_standard_detection(test_case["image"], test_case["ground_truth"])
        standard_result["test_id"] = i
        all_results.append(standard_result)
        
        # Test advanced detection
        advanced_result = tester.test_advanced_detection(test_case["image"], test_case["ground_truth"])
        advanced_result["test_id"] = i
        all_results.append(advanced_result)
    
    # Compute metrics
    metrics = tester.compute_metrics(all_results)
    
    # Save results
    results_file = os.path.join(args.output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "metrics": metrics,
            "individual_results": all_results,
            "test_cases": [{"test_id": tc["test_id"], "ground_truth": tc["ground_truth"]} for tc in test_cases]
        }, f, indent=2)
    
    # Export to Prometheus if requested
    if args.export_prometheus:
        prometheus_file = os.path.join(args.output_dir, "metrics.prom")
        tester.export_to_prometheus(metrics, prometheus_file)
        print(f"Prometheus metrics exported to {prometheus_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total tests: {metrics.get('total_tests', 0)}")
    print(f"Successful tests: {metrics.get('successful_tests', 0)}")
    print(f"Success rate: {metrics.get('success_rate', 0):.2%}")
    print(f"Exact match accuracy: {metrics.get('exact_match_accuracy', 0):.2%}")
    print(f"Within tolerance accuracy: {metrics.get('within_tolerance_accuracy', 0):.2%}")
    print(f"Average duration: {metrics.get('avg_duration_seconds', 0):.3f}s")
    print(f"Average confidence: {metrics.get('avg_confidence', 0):.3f}")
    
    if "mode_metrics" in metrics:
        print("\nPer-mode metrics:")
        for mode, mode_metrics in metrics["mode_metrics"].items():
            print(f"  {mode}:")
            print(f"    Count: {mode_metrics['count']}")
            print(f"    Exact match accuracy: {mode_metrics['exact_match_accuracy']:.2%}")
            print(f"    Average duration: {mode_metrics['avg_duration']:.3f}s")
            print(f"    Average confidence: {mode_metrics['avg_confidence']:.3f}")
    
    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
