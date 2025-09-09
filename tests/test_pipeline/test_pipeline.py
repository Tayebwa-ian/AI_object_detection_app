#!/usr/bin/env python3
"""
Test AI Pipeline Components
"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from PIL import Image

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestPipeline(unittest.TestCase):
    """Test AI Pipeline functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.pipeline.pipeline.SAM')
    @patch('src.pipeline.pipeline.ResNet')
    def test_pipeline_initialization(self, mock_resnet, mock_sam):
        """Test pipeline initialization"""
        from src.pipeline.pipeline import ObjectDetectionPipeline
        
        # Mock the models
        mock_sam.return_value = MagicMock()
        mock_resnet.return_value = MagicMock()
        
        pipeline = ObjectDetectionPipeline()
        
        self.assertIsNotNone(pipeline)
        mock_sam.assert_called_once()
        mock_resnet.assert_called_once()
    
    @patch('src.pipeline.pipeline.SAM')
    @patch('src.pipeline.pipeline.ResNet')
    def test_process_image(self, mock_resnet, mock_sam):
        """Test image processing"""
        from src.pipeline.pipeline import ObjectDetectionPipeline
        
        # Mock the models and their methods
        mock_sam_instance = MagicMock()
        mock_sam_instance.segment.return_value = {
            'masks': np.array([[[True, False], [False, True]]]),
            'scores': [0.9, 0.8]
        }
        mock_sam.return_value = mock_sam_instance
        
        mock_resnet_instance = MagicMock()
        mock_resnet_instance.classify.return_value = {
            'predictions': ['car', 'person'],
            'confidences': [0.95, 0.87]
        }
        mock_resnet.return_value = mock_resnet_instance
        
        pipeline = ObjectDetectionPipeline()
        result = pipeline.process_image(self.test_image_path)
        
        self.assertIsNotNone(result)
        self.assertIn('objects', result)
        self.assertIn('counts', result)
    
    @patch('src.pipeline.pipeline.SAM')
    @patch('src.pipeline.pipeline.ResNet')
    def test_count_objects(self, mock_resnet, mock_sam):
        """Test object counting"""
        from src.pipeline.pipeline import ObjectDetectionPipeline
        
        # Mock the models
        mock_sam_instance = MagicMock()
        mock_sam_instance.segment.return_value = {
            'masks': np.array([[[True, False], [False, True]]]),
            'scores': [0.9, 0.8]
        }
        mock_sam.return_value = mock_sam_instance
        
        mock_resnet_instance = MagicMock()
        mock_resnet_instance.classify.return_value = {
            'predictions': ['car', 'car'],
            'confidences': [0.95, 0.87]
        }
        mock_resnet.return_value = mock_resnet_instance
        
        pipeline = ObjectDetectionPipeline()
        result = pipeline.count_objects(self.test_image_path, 'car')
        
        self.assertIsNotNone(result)
        self.assertIn('count', result)
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['count'], 0)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        from src.pipeline.pipeline import ObjectDetectionPipeline
        
        with patch('src.pipeline.pipeline.SAM'), \
             patch('src.pipeline.pipeline.ResNet'):
            
            pipeline = ObjectDetectionPipeline()
            
            with self.assertRaises(FileNotFoundError):
                pipeline.process_image('/nonexistent/path.jpg')
    
    @patch('src.pipeline.pipeline.SAM')
    @patch('src.pipeline.pipeline.ResNet')
    def test_empty_image_processing(self, mock_resnet, mock_sam):
        """Test processing of empty or invalid images"""
        from src.pipeline.pipeline import ObjectDetectionPipeline
        
        # Mock the models to return empty results
        mock_sam_instance = MagicMock()
        mock_sam_instance.segment.return_value = {
            'masks': np.array([]),
            'scores': []
        }
        mock_sam.return_value = mock_sam_instance
        
        mock_resnet_instance = MagicMock()
        mock_resnet_instance.classify.return_value = {
            'predictions': [],
            'confidences': []
        }
        mock_resnet.return_value = mock_resnet_instance
        
        pipeline = ObjectDetectionPipeline()
        result = pipeline.process_image(self.test_image_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['counts'], {})
        self.assertEqual(len(result['objects']), 0)


class TestFewShotService(unittest.TestCase):
    """Test Few-Shot Learning Service"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.pipeline.fewshot_service.ResNet50')
    def test_fewshot_service_initialization(self, mock_resnet):
        """Test few-shot service initialization"""
        from src.pipeline.fewshot_service import FewShotService
        
        # Mock ResNet50
        mock_resnet.return_value = MagicMock()
        
        service = FewShotService()
        
        self.assertIsNotNone(service)
        mock_resnet.assert_called_once()
    
    @patch('src.pipeline.fewshot_service.ResNet50')
    def test_register_object_type(self, mock_resnet):
        """Test registering a new object type"""
        from src.pipeline.fewshot_service import FewShotService
        
        # Mock ResNet50 and its methods
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_resnet.return_value = mock_model
        
        service = FewShotService()
        
        # Create test support images
        support_images = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f'support_{i}.jpg')
            test_image = Image.new('RGB', (50, 50), color='blue')
            test_image.save(img_path)
            support_images.append(img_path)
        
        result = service.register_object_type('CustomObject', support_images)
        
        self.assertIsNotNone(result)
        self.assertIn('object_type_id', result)
        self.assertIn('features', result)
    
    @patch('src.pipeline.fewshot_service.ResNet50')
    def test_predict_with_fewshot(self, mock_resnet):
        """Test prediction using few-shot learning"""
        from src.pipeline.fewshot_service import FewShotService
        
        # Mock ResNet50
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_resnet.return_value = mock_model
        
        service = FewShotService()
        
        # Create test image
        test_image_path = os.path.join(self.temp_dir, 'test.jpg')
        test_image = Image.new('RGB', (100, 100), color='green')
        test_image.save(test_image_path)
        
        result = service.predict_with_fewshot(test_image_path, 'CustomObject')
        
        self.assertIsNotNone(result)
        self.assertIn('count', result)
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['count'], 0)
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)


if __name__ == '__main__':
    unittest.main()
