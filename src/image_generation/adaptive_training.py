#!/usr/bin/env python3
"""
Adaptive Training System

This system automatically determines the optimal number of training images
based on object complexity, variability, and desired accuracy.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import time
import json

from tools.train_new_object import ObjectTrainer
from tools.image_generator import ImageGenerator

class AdaptiveTrainer(ObjectTrainer):
    """
    Adaptive trainer that automatically determines optimal training parameters.
    """
    
    def __init__(self, api_base: str = "http://127.0.0.1:5000"):
        super().__init__(api_base)
        
        # Training strategies based on object type
        self.training_strategies = {
            "simple": {
                "min_images": 5,
                "max_images": 15,
                "iterations": 3,
                "target_accuracy": 0.8,
                "description": "Simple geometric shapes, basic objects"
            },
            "moderate": {
                "min_images": 10,
                "max_images": 25,
                "iterations": 5,
                "target_accuracy": 0.85,
                "description": "Faces, vehicles, common objects"
            },
            "complex": {
                "min_images": 20,
                "max_images": 50,
                "iterations": 8,
                "target_accuracy": 0.9,
                "description": "Animals, detailed scenes, textured objects"
            }
        }
    
    def analyze_object_complexity(self, object_name: str, example_images: Optional[List] = None) -> str:
        """
        Analyze object complexity to determine training strategy.
        
        Returns: "simple", "moderate", or "complex"
        """
        print(f" Analyzing complexity of '{object_name}'...")
        
        # Heuristics for complexity detection
        complexity_indicators = {
            "simple": [
                "circle", "square", "rectangle", "triangle", "line", "dot",
                "ball", "box", "cube", "sphere", "cylinder"
            ],
            "moderate": [
                "face", "person", "car", "bike", "phone", "book", "chair",
                "table", "door", "window", "tree", "house", "building"
            ],
            "complex": [
                "animal", "dog", "cat", "bird", "fish", "elephant", "tiger",
                "landscape", "forest", "mountain", "ocean", "sky", "cloud",
                "texture", "pattern", "art", "painting", "sculpture"
            ]
        }
        
        object_lower = object_name.lower()
        
        # Check against complexity indicators
        for complexity, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in object_lower:
                    print(f"   📊 Detected as '{complexity}': {indicator} match")
                    return complexity
        
        # Default to moderate if no clear match
        print(f"    Defaulting to 'moderate' complexity")
        return "moderate"
    
    def calculate_optimal_image_count(self, object_name: str, example_images: Optional[List] = None) -> Dict:
        """
        Calculate optimal number of training images based on analysis.
        """
        complexity = self.analyze_object_complexity(object_name, example_images)
        strategy = self.training_strategies[complexity]
        
        # Base calculation
        base_count = strategy["min_images"]
        
        # Adjust based on example images quality
        if example_images:
            # More examples = can start with fewer generated images
            example_count = len(example_images)
            if example_count >= 5:
                base_count = max(strategy["min_images"], base_count - 2)
            elif example_count >= 10:
                base_count = max(strategy["min_images"], base_count - 5)
        
        # Adjust based on object name length (proxy for specificity)
        if len(object_name) > 10:  # More specific names might need more examples
            base_count += 5
        
        # Ensure within bounds
        optimal_count = max(strategy["min_images"], 
                           min(strategy["max_images"], base_count))
        
        return {
            "complexity": complexity,
            "strategy": strategy,
            "optimal_count": optimal_count,
            "iterations": strategy["iterations"],
            "target_accuracy": strategy["target_accuracy"]
        }
    
    def adaptive_training_loop(self, object_name: str, example_images: Optional[List] = None) -> Dict:
        """
        Run adaptive training that adjusts based on performance.
        """
        print(f" Starting Adaptive Training for '{object_name}'")
        print("=" * 60)
        
        # Calculate initial parameters
        params = self.calculate_optimal_image_count(object_name, example_images)
        print(f" Training Strategy: {params['complexity']}")
        print(f"   Description: {params['strategy']['description']}")
        print(f"   Target Accuracy: {params['target_accuracy']:.1%}")
        print(f"   Initial Image Count: {params['optimal_count']}")
        print(f"   Max Iterations: {params['iterations']}")
        
        # Step 1: Initial registration
        print(f"\n Step 1: Initial Registration")
        reg_result = self.register_object(object_name, example_images)
        if not reg_result.get('success'):
            return {"success": False, "error": "Initial registration failed"}
        
        # Step 2: Adaptive training loop
        print(f"\n Step 2: Adaptive Training Loop")
        training_history = []
        current_image_count = params['optimal_count']
        
        for iteration in range(params['iterations']):
            print(f"\n--- Training Round {iteration + 1}/{params['iterations']} ---")
            print(f"   Generating {current_image_count} images...")
            
            # Generate training images
            training_images = self.generate_training_images(object_name, current_image_count)
            
            # Train with generated images
            train_result = self.train_with_images(object_name, training_images)
            training_history.append({
                'iteration': iteration + 1,
                'image_count': current_image_count,
                'result': train_result
            })
            
            if not train_result.get('success'):
                print(f"    Round {iteration + 1} failed, continuing...")
                continue
            
            # Test current performance
            print(f"    Testing current performance...")
            test_images = self.generate_training_images(object_name, 5)
            test_result = self.test_object_detection(object_name, test_images)
            
            current_accuracy = test_result['success_rate']
            current_confidence = test_result['avg_confidence']
            
            print(f"    Current Accuracy: {current_accuracy:.1%}")
            print(f"    Current Confidence: {current_confidence:.3f}")
            
            # Check if we've reached target
            if current_accuracy >= params['target_accuracy'] and current_confidence >= 0.7:
                print(f"    Target accuracy reached! Stopping early.")
                break
            
            # Adaptive adjustment for next round
            if current_accuracy < 0.5:  # Very low accuracy
                current_image_count = min(params['strategy']['max_images'], 
                                        current_image_count + 10)
                print(f"    Low accuracy detected, increasing to {current_image_count} images")
            elif current_accuracy < 0.7:  # Moderate accuracy
                current_image_count = min(params['strategy']['max_images'], 
                                        current_image_count + 5)
                print(f"    Moderate accuracy, increasing to {current_image_count} images")
            else:  # Good accuracy, maintain or slightly increase
                current_image_count = min(params['strategy']['max_images'], 
                                        current_image_count + 2)
                print(f"    Good accuracy, maintaining {current_image_count} images")
            
            # Small delay between rounds
            time.sleep(1)
        
        # Step 3: Final comprehensive test
        print(f"\n Step 3: Final Comprehensive Test")
        final_test_images = self.generate_training_images(object_name, 10)
        final_test_result = self.test_object_detection(object_name, final_test_images)
        
        # Calculate final metrics
        successful_rounds = sum(1 for h in training_history if h['result'].get('success'))
        total_images_used = sum(h['image_count'] for h in training_history)
        
        print(f"\n Final Results for '{object_name}':")
        print(f"   Final Accuracy: {final_test_result['success_rate']:.1%}")
        print(f"   Final Confidence: {final_test_result['avg_confidence']:.3f}")
        print(f"   Training Rounds: {len(training_history)}")
        print(f"   Successful Rounds: {successful_rounds}")
        print(f"   Total Images Used: {total_images_used}")
        print(f"   Target Met: {'Yes' if final_test_result['success_rate'] >= params['target_accuracy'] else 'No'}")
        
        return {
            "success": True,
            "object_name": object_name,
            "complexity": params['complexity'],
            "strategy": params['strategy'],
            "final_accuracy": final_test_result['success_rate'],
            "final_confidence": final_test_result['avg_confidence'],
            "training_history": training_history,
            "total_images_used": total_images_used,
            "target_met": final_test_result['success_rate'] >= params['target_accuracy']
        }
    
    def recommend_training_parameters(self, object_name: str) -> Dict:
        """
        Provide training recommendations without actually training.
        """
        params = self.calculate_optimal_image_count(object_name)
        
        print(f" Training Recommendations for '{object_name}':")
        print(f"   Complexity: {params['complexity']}")
        print(f"   Recommended Images: {params['optimal_count']}")
        print(f"   Recommended Iterations: {params['iterations']}")
        print(f"   Target Accuracy: {params['target_accuracy']:.1%}")
        print(f"   Description: {params['strategy']['description']}")
        
        return params

def main():
    """Test the adaptive training system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive object training")
    parser.add_argument("--object", required=True, help="Object to train")
    parser.add_argument("--examples", help="Path to example images")
    parser.add_argument("--recommend-only", action="store_true", 
                       help="Only show recommendations, don't train")
    
    args = parser.parse_args()
    
    trainer = AdaptiveTrainer()
    
    if not trainer.check_api_health():
        print(" API is not running!")
        return
    
    if args.recommend_only:
        trainer.recommend_training_parameters(args.object)
    else:
        example_images = None
        if args.examples and os.path.exists(args.examples):
            example_images = [str(p) for p in Path(args.examples).glob("*.jpg")]
        
        result = trainer.adaptive_training_loop(args.object, example_images)
        
        if result.get('success'):
            print(f"\npleas Adaptive training completed!")
            print(f"   Final accuracy: {result['final_accuracy']:.1%}")
            print(f"   Target met: {'Yes' if result['target_met'] else 'No'}")
        else:
            print(f"\n Training failed: {result.get('error')}")

if __name__ == "__main__":
    main()
