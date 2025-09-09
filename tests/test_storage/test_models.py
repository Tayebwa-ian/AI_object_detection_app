#!/usr/bin/env python3
"""
Test Storage Models
"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.storage.inputs import Input
from src.storage.object_types import ObjectType
from src.storage.outputs import Output
from src.storage.fewshot_models import FewShotObjectType, FewShotPrediction


class TestInputModel(unittest.TestCase):
    """Test Input model"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = {
            'id': 'test_input_id',
            'description': 'Test input',
            'image_path': '/test/path.jpg',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    
    def test_input_creation(self):
        """Test Input model creation"""
        input_obj = Input(**self.test_data)
        
        self.assertEqual(input_obj.id, 'test_input_id')
        self.assertEqual(input_obj.description, 'Test input')
        self.assertEqual(input_obj.image_path, '/test/path.jpg')
        self.assertIsNotNone(input_obj.created_at)
        self.assertIsNotNone(input_obj.updated_at)
    
    def test_input_to_dict(self):
        """Test Input to_dict method"""
        input_obj = Input(**self.test_data)
        result = input_obj.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'test_input_id')
        self.assertEqual(result['description'], 'Test input')
        self.assertEqual(result['image_path'], '/test/path.jpg')
    
    def test_input_save(self):
        """Test Input save method"""
        input_obj = Input(**self.test_data)
        
        with patch('src.storage.inputs.database') as mock_db:
            mock_db.save.return_value = input_obj
            result = input_obj.save()
            
            mock_db.save.assert_called_once_with(input_obj)
            self.assertEqual(result, input_obj)


class TestObjectTypeModel(unittest.TestCase):
    """Test ObjectType model"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = {
            'id': 'test_obj_type_id',
            'name': 'Car',
            'description': 'A vehicle',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    
    def test_object_type_creation(self):
        """Test ObjectType model creation"""
        obj_type = ObjectType(**self.test_data)
        
        self.assertEqual(obj_type.id, 'test_obj_type_id')
        self.assertEqual(obj_type.name, 'Car')
        self.assertEqual(obj_type.description, 'A vehicle')
        self.assertIsNotNone(obj_type.created_at)
        self.assertIsNotNone(obj_type.updated_at)
    
    def test_object_type_to_dict(self):
        """Test ObjectType to_dict method"""
        obj_type = ObjectType(**self.test_data)
        result = obj_type.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'test_obj_type_id')
        self.assertEqual(result['name'], 'Car')
        self.assertEqual(result['description'], 'A vehicle')
    
    def test_object_type_save(self):
        """Test ObjectType save method"""
        obj_type = ObjectType(**self.test_data)
        
        with patch('src.storage.object_types.database') as mock_db:
            mock_db.save.return_value = obj_type
            result = obj_type.save()
            
            mock_db.save.assert_called_once_with(obj_type)
            self.assertEqual(result, obj_type)


class TestOutputModel(unittest.TestCase):
    """Test Output model"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = {
            'id': 'test_output_id',
            'predicted_count': 5,
            'pred_confidence': 0.95,
            'object_type_id': 'test_obj_type_id',
            'input_id': 'test_input_id',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    
    def test_output_creation(self):
        """Test Output model creation"""
        output = Output(**self.test_data)
        
        self.assertEqual(output.id, 'test_output_id')
        self.assertEqual(output.predicted_count, 5)
        self.assertEqual(output.pred_confidence, 0.95)
        self.assertEqual(output.object_type_id, 'test_obj_type_id')
        self.assertEqual(output.input_id, 'test_input_id')
        self.assertIsNotNone(output.created_at)
        self.assertIsNotNone(output.updated_at)
    
    def test_output_to_dict(self):
        """Test Output to_dict method"""
        output = Output(**self.test_data)
        result = output.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'test_output_id')
        self.assertEqual(result['predicted_count'], 5)
        self.assertEqual(result['pred_confidence'], 0.95)
        self.assertEqual(result['object_type_id'], 'test_obj_type_id')
        self.assertEqual(result['input_id'], 'test_input_id')
    
    def test_output_save(self):
        """Test Output save method"""
        output = Output(**self.test_data)
        
        with patch('src.storage.outputs.database') as mock_db:
            mock_db.save.return_value = output
            result = output.save()
            
            mock_db.save.assert_called_once_with(output)
            self.assertEqual(result, output)


class TestFewShotModels(unittest.TestCase):
    """Test FewShot models"""
    
    def setUp(self):
        """Set up test data"""
        self.fewshot_obj_data = {
            'id': 'test_fewshot_id',
            'name': 'CustomObject',
            'description': 'A custom object type',
            'support_images': ['img1.jpg', 'img2.jpg'],
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        self.fewshot_pred_data = {
            'id': 'test_pred_id',
            'fewshot_object_type_id': 'test_fewshot_id',
            'input_id': 'test_input_id',
            'predicted_count': 3,
            'confidence': 0.88,
            'created_at': datetime.now()
        }
    
    def test_fewshot_object_type_creation(self):
        """Test FewShotObjectType model creation"""
        fewshot_obj = FewShotObjectType(**self.fewshot_obj_data)
        
        self.assertEqual(fewshot_obj.id, 'test_fewshot_id')
        self.assertEqual(fewshot_obj.name, 'CustomObject')
        self.assertEqual(fewshot_obj.description, 'A custom object type')
        self.assertEqual(fewshot_obj.support_images, ['img1.jpg', 'img2.jpg'])
        self.assertIsNotNone(fewshot_obj.created_at)
        self.assertIsNotNone(fewshot_obj.updated_at)
    
    def test_fewshot_prediction_creation(self):
        """Test FewShotPrediction model creation"""
        fewshot_pred = FewShotPrediction(**self.fewshot_pred_data)
        
        self.assertEqual(fewshot_pred.id, 'test_pred_id')
        self.assertEqual(fewshot_pred.fewshot_object_type_id, 'test_fewshot_id')
        self.assertEqual(fewshot_pred.input_id, 'test_input_id')
        self.assertEqual(fewshot_pred.predicted_count, 3)
        self.assertEqual(fewshot_pred.confidence, 0.88)
        self.assertIsNotNone(fewshot_pred.created_at)


if __name__ == '__main__':
    unittest.main()

