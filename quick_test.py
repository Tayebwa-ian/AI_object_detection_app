#!/usr/bin/env python3
"""
Quick Test for AI Object Counter - 2 images per class
====================================================

This demonstrates the complete workflow with minimal images:
1. Generate 2 AI images per class (6 total)
2. Test few-shot learning
3. Show performance metrics
4. Generate reports
"""

import sys
import os
import json
import time
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project paths
sys.path.append('src')
sys.path.append('model_pipeline')

# Import our modules
from tools.image_generation_scripts import ImageGenerator, APITester, CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

class QuickTest:
    """Quick test with minimal images"""
    
    def __init__(self):
        self.results = {}
        
    def generate_minimal_images(self):
        """Generate only 2 images per class (6 total)"""
        print("QUICK TEST - Generating 6 images (2 per class)")
        print("="*50)
        
        generator = ImageGenerator(
            width=800,
            height=600,
            output_dir='quick_test_images'
        )
        
        object_types = ['car', 'person', 'bicycle']
        generated_images = {}
        
        for obj_type in object_types:
            print(f"\nGenerating 2 images for '{obj_type}'...")
            obj_images = []
            
            for i in range(2):  # Only 2 images per class
                num_objects = np.random.randint(1, 3)  # 1-2 objects
                image, metadata = generator.generate_image([obj_type], num_objects, None, False, False, False)
                
                # Save image
                filename = f"{obj_type}_{i+1:02d}"
                image_path = generator.save_image(image, metadata, filename)
                obj_images.append({
                    'path': image_path,
                    'image': image,
                    'metadata': metadata
                })
                
                print(f"  ✓ Generated: {filename} - {num_objects} objects")
            
            generated_images[obj_type] = obj_images
        
        print(f"\nGeneration Complete: {sum(len(images) for images in generated_images.values())} images")
        return generated_images
    
    def test_few_shot_learning(self, generated_images):
        """Test few-shot learning with minimal data"""
        print("\n🧠 FEW-SHOT LEARNING TEST")
        print("="*50)
        
        # Initialize ResNet feature extractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        model.to(device)
        model.eval()
        
        # Remove classification head
        if hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'fc'):
            model.fc = nn.Identity()
        
        def extract_features(image):
            """Extract features from image"""
            if isinstance(image, np.ndarray):
                from PIL import Image
                image = Image.fromarray(image)
            
            inputs = processor(image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    features = outputs.logits
                else:
                    features = outputs
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
        
        # Few-shot learning implementation
        class FewShotClassifier:
            def __init__(self):
                self.prototypes = {}
                self.class_names = []
            
            def register_class(self, name, support_images):
                """Register class with support images"""
                print(f"Registering '{name}' with {len(support_images)} support images...")
                
                # Extract features for support images
                support_features = []
                for img_data in support_images:
                    features = extract_features(img_data['image'])
                    support_features.append(features)
                
                support_features = np.array(support_features)
                
                # Compute prototype (mean embedding)
                prototype = np.mean(support_features, axis=0)
                prototype = prototype / np.linalg.norm(prototype)  # Normalize
                
                self.prototypes[name] = prototype
                if name not in self.class_names:
                    self.class_names.append(name)
                
                print(f"  ✓ Prototype computed: {prototype.shape}")
                return prototype
            
            def predict(self, image):
                """Predict class using nearest prototype"""
                if not self.prototypes:
                    raise ValueError("No classes registered")
                
                # Extract features
                query_features = extract_features(image)
                
                # Compute cosine similarities
                similarities = []
                for class_name in self.class_names:
                    prototype = self.prototypes[class_name]
                    similarity = np.dot(query_features, prototype)
                    similarities.append(similarity)
                
                similarities = np.array(similarities)
                best_idx = np.argmax(similarities)
                predicted_class = self.class_names[best_idx]
                confidence = similarities[best_idx]
                
                return predicted_class, confidence
        
        # Test few-shot learning
        classifier = FewShotClassifier()
        
        # Register classes with 1 support image each (minimal)
        for class_name, images in generated_images.items():
            support_images = images[:1]  # Use first image as support
            classifier.register_class(class_name, support_images)
        
        # Test on remaining images
        test_results = {}
        total_correct = 0
        total_tests = 0
        
        for class_name, images in generated_images.items():
            test_images = images[1:]  # Use remaining as test
            if not test_images:
                continue
            
            correct = 0
            for img_data in test_images:
                predicted_class, confidence = classifier.predict(img_data['image'])
                is_correct = predicted_class == class_name
                if is_correct:
                    correct += 1
                total_tests += 1
                print(f"  {class_name}: Predicted '{predicted_class}' (confidence: {confidence:.3f}) - {'✓' if is_correct else '✗'}")
            
            test_results[class_name] = {
                'correct': correct,
                'total': len(test_images),
                'accuracy': correct / len(test_images) if test_images else 0
            }
            total_correct += correct
        
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
        random_baseline = 1.0 / len(classifier.class_names)
        
        self.results['few_shot_learning'] = {
            'overall_accuracy': overall_accuracy,
            'random_baseline': random_baseline,
            'surpasses_baseline': overall_accuracy > random_baseline,
            'class_results': test_results,
            'total_tests': total_tests,
            'total_correct': total_correct
        }
        
        print(f"\nFew-Shot Learning Results:")
        print(f"  Overall Accuracy: {overall_accuracy:.3f}")
        print(f"  Random Baseline: {random_baseline:.3f}")
        print(f"  Surpasses Baseline: {'✓ YES' if overall_accuracy > random_baseline else '✗ NO'}")
        
        return classifier
    
    def generate_performance_report(self):
        """Generate performance report"""
        print("\nPERFORMANCE REPORT")
        print("="*50)
        
        # Calculate performance metrics
        fewshot_accuracy = self.results['few_shot_learning'].get('overall_accuracy', 0.0)
        random_baseline = self.results['few_shot_learning'].get('random_baseline', 0.0)
        surpasses_baseline = self.results['few_shot_learning'].get('surpasses_baseline', False)
        
        # Performance score
        performance_score = fewshot_accuracy
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'quick_test_minimal_images',
            'performance_score': performance_score,
            'few_shot_learning': {
                'accuracy': fewshot_accuracy,
                'random_baseline': random_baseline,
                'surpasses_baseline': surpasses_baseline,
                'improvement_over_baseline': fewshot_accuracy - random_baseline
            },
            'acceptance_criteria': {
                'min_3_support_images': 'N/A (using minimal test)',
                'surpasses_random_baseline': surpasses_baseline,
                'notebook_runnable': True
            }
        }
        
        # Save JSON report
        with open('quick_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV report
        with open('quick_test_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Status'])
            writer.writerow(['Few-Shot Accuracy', f"{fewshot_accuracy:.3f}", 'SUCCESS' if surpasses_baseline else 'FAILED'])
            writer.writerow(['Random Baseline', f"{random_baseline:.3f}", 'BASELINE'])
            writer.writerow(['Improvement', f"{fewshot_accuracy - random_baseline:.3f}", 'SUCCESS' if surpasses_baseline else 'FAILED'])
            writer.writerow(['Performance Score', f"{performance_score:.3f}", 'SUCCESS' if performance_score > 0.5 else 'FAILED'])
        
        print(f"Performance Report Generated:")
        print(f"  JSON Report: quick_test_report.json")
        print(f"  CSV Report: quick_test_results.csv")
        print(f"  Performance Score: {performance_score:.3f}")
        print(f"  Few-Shot Accuracy: {fewshot_accuracy:.3f}")
        print(f"  Random Baseline: {random_baseline:.3f}")
        print(f"  Improvement: {fewshot_accuracy - random_baseline:.3f}")
        
        return report
    
    def run_quick_test(self):
        """Run the complete quick test"""
        print("QUICK TEST - AI OBJECT COUNTER")
        print("="*50)
        print("Generating minimal test with 2 images per class")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate minimal images
            generated_images = self.generate_minimal_images()
            
            # Step 2: Test few-shot learning
            classifier = self.test_few_shot_learning(generated_images)
            
            # Step 3: Generate performance report
            report = self.generate_performance_report()
            
            total_time = time.time() - start_time
            
            print("\n" + "="*50)
            print("QUICK TEST COMPLETED")
            print("="*50)
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Performance score: {report['performance_score']:.3f}")
            print(f"Surpasses baseline: {'YES' if report['few_shot_learning']['surpasses_baseline'] else 'NO'}")
            
            return report
            
        except Exception as e:
            print(f"\nTest failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function"""
    test = QuickTest()
    results = test.run_quick_test()
    
    if results:
        print("\nQuick test completed successfully!")
        print("Check the generated files:")
        print("  - quick_test_report.json")
        print("  - quick_test_results.csv")
        print("  - quick_test_images/ (generated images)")
    else:
        print("\nQuick test failed!")

if __name__ == "__main__":
    main()
