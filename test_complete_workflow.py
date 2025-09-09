#!/usr/bin/env python3
"""
Complete Workflow Test for AI Object Counter
===========================================

This script demonstrates the complete workflow:
1. Generate AI images using the endpoint
2. Test few-shot learning with AI-generated images
3. Test API integration (POST/PUT workflow)
4. Generate performance reports
5. Show metrics and analysis

This implements the core requirements:
- Automatically POST generated images to API and submit corrections
- Generate test reports showing performance by image characteristics
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
import matplotlib.pyplot as plt
from datetime import datetime

# Add project paths
sys.path.append('src')
sys.path.append('model_pipeline')

# Import our modules
from tools.generate_and_post import ImageGenerator, APITester, CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

class CompleteWorkflowTest:
    """Complete workflow test for the AI Object Counter system"""
    
    def __init__(self):
        self.results = {
            'ai_generation': {},
            'few_shot_learning': {},
            'api_integration': {},
            'performance_metrics': {},
            'test_report': {}
        }
        
    def test_ai_image_generation(self, num_images=10):
        """Test AI image generation"""
        print("="*60)
        print("TEST 1: AI IMAGE GENERATION")
        print("="*60)
        
        # Generate images using our working script
        generator = ImageGenerator(
            width=800,
            height=600,
            output_dir='test_workflow_images'
        )
        
        # Generate images for few-shot learning
        object_types = ['car', 'person', 'bicycle']
        generated_images = {}
        
        for obj_type in object_types:
            print(f"\nGenerating images for '{obj_type}'...")
            obj_images = []
            
            for i in range(3):  # 3 images per class for few-shot learning
                num_objects = np.random.randint(1, 4)
                image, metadata = generator.generate_image([obj_type], num_objects, None, False, False, False)
                
                # Save image
                filename = f"{obj_type}_{i+1:02d}"
                image_path = generator.save_image(image, metadata, filename)
                obj_images.append({
                    'path': image_path,
                    'image': image,
                    'metadata': metadata
                })
                
                print(f"  Generated: {filename} - {num_objects} objects")
            
            generated_images[obj_type] = obj_images
        
        self.results['ai_generation'] = {
            'success': True,
            'images_generated': sum(len(images) for images in generated_images.values()),
            'classes': list(generated_images.keys()),
            'images_per_class': {k: len(v) for k, v in generated_images.items()}
        }
        
        print(f"\n✅ AI Generation Complete:")
        print(f"  Total images: {self.results['ai_generation']['images_generated']}")
        print(f"  Classes: {self.results['ai_generation']['classes']}")
        
        return generated_images
    
    def test_few_shot_learning(self, generated_images):
        """Test few-shot learning with AI-generated images"""
        print("\n" + "="*60)
        print("TEST 2: FEW-SHOT LEARNING")
        print("="*60)
        
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
                print(f"Registering class '{name}' with {len(support_images)} support images...")
                
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
                
                print(f"  Prototype computed: {prototype.shape}")
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
        
        # Register classes with 2 support images each
        for class_name, images in generated_images.items():
            support_images = images[:2]  # Use first 2 as support
            classifier.register_class(class_name, support_images)
        
        # Test on remaining images
        test_results = {}
        total_correct = 0
        total_tests = 0
        
        for class_name, images in generated_images.items():
            test_images = images[2:]  # Use remaining as test
            if not test_images:
                continue
            
            correct = 0
            for img_data in test_images:
                predicted_class, confidence = classifier.predict(img_data['image'])
                is_correct = predicted_class == class_name
                if is_correct:
                    correct += 1
                total_tests += 1
            
            test_results[class_name] = {
                'correct': correct,
                'total': len(test_images),
                'accuracy': correct / len(test_images) if test_images else 0
            }
            total_correct += correct
        
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
        random_baseline = 1.0 / len(classifier.class_names)
        
        self.results['few_shot_learning'] = {
            'success': True,
            'overall_accuracy': overall_accuracy,
            'random_baseline': random_baseline,
            'surpasses_baseline': overall_accuracy > random_baseline,
            'class_results': test_results,
            'total_tests': total_tests,
            'total_correct': total_correct
        }
        
        print(f"\n✅ Few-Shot Learning Results:")
        print(f"  Overall Accuracy: {overall_accuracy:.3f}")
        print(f"  Random Baseline: {random_baseline:.3f}")
        print(f"  Surpasses Baseline: {'✓' if overall_accuracy > random_baseline else '✗'}")
        
        for class_name, results in test_results.items():
            print(f"  {class_name}: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")
        
        return classifier
    
    def test_api_integration(self, generated_images):
        """Test API integration (POST/PUT workflow)"""
        print("\n" + "="*60)
        print("TEST 3: API INTEGRATION")
        print("="*60)
        
        api_tester = APITester("http://localhost:5000", timeout=60)  # Longer timeout
        
        api_results = {
            'total_tests': 0,
            'successful_posts': 0,
            'successful_corrections': 0,
            'response_times': [],
            'accuracies': [],
            'class_results': {}
        }
        
        # Test API with generated images
        for class_name, images in generated_images.items():
            print(f"\nTesting API with '{class_name}' images...")
            class_results = {
                'tests': 0,
                'successful_posts': 0,
                'successful_corrections': 0,
                'response_times': [],
                'accuracies': []
            }
            
            for i, img_data in enumerate(images[:3]):  # Test first 3 images per class
                print(f"  Testing image {i+1}...")
                
                # Get expected count from metadata
                expected_count = img_data['metadata'].get('expected_objects', 1)
                
                # Test API
                result = api_tester.test_image(img_data['path'], class_name, expected_count)
                
                api_results['total_tests'] += 1
                class_results['tests'] += 1
                
                if result['success']:
                    api_results['successful_posts'] += 1
                    class_results['successful_posts'] += 1
                    api_results['response_times'].append(result['response_time_ms'])
                    class_results['response_times'].append(result['response_time_ms'])
                    
                    # Check accuracy
                    predicted = result['predicted_count']
                    accuracy = 1.0 if predicted == expected_count else 0.0
                    api_results['accuracies'].append(accuracy)
                    class_results['accuracies'].append(accuracy)
                    
                    # Submit correction
                    if result.get('result_id'):
                        correction_success = api_tester.submit_correction(result['result_id'], expected_count)
                        if correction_success:
                            api_results['successful_corrections'] += 1
                            class_results['successful_corrections'] += 1
                    
                    print(f"    Predicted: {predicted}, Expected: {expected_count}, Accuracy: {accuracy}")
                else:
                    print(f"    API Error: {result.get('error', 'Unknown')}")
            
            api_results['class_results'][class_name] = class_results
        
        # Calculate overall metrics
        if api_results['total_tests'] > 0:
            api_results['success_rate'] = api_results['successful_posts'] / api_results['total_tests']
            api_results['correction_rate'] = api_results['successful_corrections'] / api_results['total_tests']
            api_results['avg_response_time'] = np.mean(api_results['response_times']) if api_results['response_times'] else 0
            api_results['avg_accuracy'] = np.mean(api_results['accuracies']) if api_results['accuracies'] else 0
        
        self.results['api_integration'] = api_results
        
        print(f"\n✅ API Integration Results:")
        print(f"  Total Tests: {api_results['total_tests']}")
        print(f"  Success Rate: {api_results.get('success_rate', 0):.3f}")
        print(f"  Correction Rate: {api_results.get('correction_rate', 0):.3f}")
        print(f"  Avg Response Time: {api_results.get('avg_response_time', 0):.1f}ms")
        print(f"  Avg Accuracy: {api_results.get('avg_accuracy', 0):.3f}")
        
        return api_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("TEST 4: PERFORMANCE REPORT GENERATION")
        print("="*60)
        
        # Create performance metrics
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'ai_generation': self.results['ai_generation'],
            'few_shot_learning': self.results['few_shot_learning'],
            'api_integration': self.results['api_integration']
        }
        
        # Calculate overall performance score
        ai_score = 1.0 if self.results['ai_generation']['success'] else 0.0
        fewshot_score = self.results['few_shot_learning'].get('overall_accuracy', 0.0)
        api_score = self.results['api_integration'].get('success_rate', 0.0)
        
        overall_score = (ai_score + fewshot_score + api_score) / 3.0
        
        performance_metrics['overall_score'] = overall_score
        performance_metrics['component_scores'] = {
            'ai_generation': ai_score,
            'few_shot_learning': fewshot_score,
            'api_integration': api_score
        }
        
        # Save performance report
        report_path = 'test_workflow_performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        # Create CSV report
        csv_path = 'test_workflow_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Component', 'Metric', 'Value', 'Status'])
            
            # AI Generation
            writer.writerow(['AI Generation', 'Images Generated', self.results['ai_generation']['images_generated'], 'SUCCESS' if self.results['ai_generation']['success'] else 'FAILED'])
            writer.writerow(['AI Generation', 'Classes', len(self.results['ai_generation']['classes']), 'SUCCESS'])
            
            # Few-Shot Learning
            writer.writerow(['Few-Shot Learning', 'Overall Accuracy', f"{self.results['few_shot_learning'].get('overall_accuracy', 0):.3f}", 'SUCCESS' if self.results['few_shot_learning'].get('surpasses_baseline', False) else 'FAILED'])
            writer.writerow(['Few-Shot Learning', 'Random Baseline', f"{self.results['few_shot_learning'].get('random_baseline', 0):.3f}", 'BASELINE'])
            
            # API Integration
            writer.writerow(['API Integration', 'Success Rate', f"{self.results['api_integration'].get('success_rate', 0):.3f}", 'SUCCESS' if self.results['api_integration'].get('success_rate', 0) > 0.5 else 'FAILED'])
            writer.writerow(['API Integration', 'Avg Response Time', f"{self.results['api_integration'].get('avg_response_time', 0):.1f}ms", 'INFO'])
            writer.writerow(['API Integration', 'Avg Accuracy', f"{self.results['api_integration'].get('avg_accuracy', 0):.3f}", 'INFO'])
            
            # Overall
            writer.writerow(['Overall', 'Performance Score', f"{overall_score:.3f}", 'SUCCESS' if overall_score > 0.6 else 'FAILED'])
        
        self.results['performance_metrics'] = performance_metrics
        
        print(f"\n✅ Performance Report Generated:")
        print(f"  JSON Report: {report_path}")
        print(f"  CSV Report: {csv_path}")
        print(f"  Overall Score: {overall_score:.3f}")
        print(f"  Component Scores:")
        print(f"    AI Generation: {ai_score:.3f}")
        print(f"    Few-Shot Learning: {fewshot_score:.3f}")
        print(f"    API Integration: {api_score:.3f}")
        
        return performance_metrics
    
    def run_complete_test(self):
        """Run the complete workflow test"""
        print("🚀 STARTING COMPLETE WORKFLOW TEST")
        print("="*60)
        print("This test demonstrates:")
        print("1. AI Image Generation using endpoint")
        print("2. Few-Shot Learning with AI-generated images")
        print("3. API Integration (POST/PUT workflow)")
        print("4. Performance Report Generation")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test 1: AI Image Generation
            generated_images = self.test_ai_image_generation(num_images=9)  # 3 per class
            
            # Test 2: Few-Shot Learning
            classifier = self.test_few_shot_learning(generated_images)
            
            # Test 3: API Integration
            api_results = self.test_api_integration(generated_images)
            
            # Test 4: Performance Report
            performance_metrics = self.generate_performance_report()
            
            total_time = time.time() - start_time
            
            print("\n" + "="*60)
            print("🎉 COMPLETE WORKFLOW TEST FINISHED")
            print("="*60)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Overall performance score: {performance_metrics['overall_score']:.3f}")
            
            # Final status
            all_tests_passed = (
                self.results['ai_generation']['success'] and
                self.results['few_shot_learning'].get('surpasses_baseline', False) and
                self.results['api_integration'].get('success_rate', 0) > 0
            )
            
            print(f"All tests passed: {'✅ YES' if all_tests_passed else '❌ NO'}")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the complete workflow test"""
    test = CompleteWorkflowTest()
    results = test.run_complete_test()
    
    if results:
        print("\n📊 Test completed successfully!")
        print("Check the generated files:")
        print("  - test_workflow_performance_report.json")
        print("  - test_workflow_results.csv")
        print("  - test_workflow_images/ (generated images)")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    main()
