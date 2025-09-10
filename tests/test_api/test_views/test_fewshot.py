#!/usr/bin/env python3
"""
Test Few-Shot Learning API Views
"""
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_restful import Api
from marshmallow import ValidationError

from src.api.views.fewshot import (
    FewShotRegister, FewShotCount, FewShotObjectTypes, 
    FewShotObjectTypeSingle, FewShotPredictions
)


class TestFewShotViews(unittest.TestCase):
    """Test Few-Shot Learning API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        app = Flask(__name__)
        api = Api(app)
        api.add_resource(FewShotRegister, '/api/fewshot/register')
        api.add_resource(FewShotCount, '/api/fewshot/count')
        api.add_resource(FewShotObjectTypes, '/api/fewshot/object-types')
        api.add_resource(FewShotObjectTypeSingle, '/api/fewshot/object-types/<string:obj_type_id>')
        api.add_resource(FewShotPredictions, '/api/fewshot/predictions')

        self.app = app
        self.client = app.test_client()

        # Patch dependencies
        self.fewshot_service_patcher = patch('src.api.views.fewshot.fewshot_service')
        self.fewshot_schema_patcher = patch('src.api.views.fewshot.fewshot_object_type_schema')
        self.fewshot_pred_schema_patcher = patch('src.api.views.fewshot.fewshot_prediction_schema')
        self.db_patcher = patch('src.api.views.fewshot.database')

        self.mock_fewshot_service = self.fewshot_service_patcher.start()
        self.mock_fewshot_schema = self.fewshot_schema_patcher.start()
        self.mock_fewshot_pred_schema = self.fewshot_pred_schema_patcher.start()
        self.mock_db = self.db_patcher.start()

        # Setup mock methods
        self.mock_fewshot_schema.load = MagicMock()
        self.mock_fewshot_schema.dump = MagicMock()
        self.mock_fewshot_pred_schema.dump = MagicMock()

    def tearDown(self):
        """Clean up patches"""
        patch.stopall()

    def test_register_object_type_success(self):
        """Test successful object type registration"""
        payload = {
            'name': 'CustomObject',
            'description': 'A custom object',
            'support_images': ['img1.jpg', 'img2.jpg']
        }
        
        self.mock_fewshot_schema.load.return_value = payload
        self.mock_fewshot_service.register_object_type.return_value = {
            'object_type_id': 'fs_123',
            'features': [0.1, 0.2, 0.3]
        }
        self.mock_fewshot_schema.dump.return_value = {
            'id': 'fs_123',
            'name': 'CustomObject',
            'description': 'A custom object'
        }

        resp = self.client.post('/api/fewshot/register', json=payload)
        
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertEqual(data['name'], 'CustomObject')
        self.mock_fewshot_service.register_object_type.assert_called_once()

    def test_register_object_type_validation_error(self):
        """Test registration with validation error"""
        payload = {'name': ''}  # Invalid payload
        self.mock_fewshot_schema.load.side_effect = ValidationError({'name': ['required']})

        resp = self.client.post('/api/fewshot/register', json=payload)
        
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data['status'], 'fail')

    def test_fewshot_count_success(self):
        """Test successful few-shot counting"""
        payload = {
            'image_path': '/test/image.jpg',
            'object_type_name': 'CustomObject'
        }
        
        self.mock_fewshot_service.predict_with_fewshot.return_value = {
            'count': 3,
            'confidence': 0.92
        }
        self.mock_fewshot_pred_schema.dump.return_value = {
            'count': 3,
            'confidence': 0.92,
            'object_type_name': 'CustomObject'
        }

        resp = self.client.post('/api/fewshot/count', json=payload)
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['count'], 3)
        self.assertEqual(data['confidence'], 0.92)

    def test_fewshot_count_service_error(self):
        """Test few-shot counting with service error"""
        payload = {
            'image_path': '/test/image.jpg',
            'object_type_name': 'NonexistentObject'
        }
        
        self.mock_fewshot_service.predict_with_fewshot.side_effect = Exception('Object type not found')

        resp = self.client.post('/api/fewshot/count', json=payload)
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')

    def test_get_fewshot_object_types_success(self):
        """Test getting all few-shot object types"""
        mock_objects = [MagicMock(), MagicMock()]
        self.mock_db.all.return_value = mock_objects
        self.mock_fewshot_schema.dump.return_value = [
            {'id': 'fs_1', 'name': 'Object1'},
            {'id': 'fs_2', 'name': 'Object2'}
        ]

        resp = self.client.get('/api/fewshot/object-types')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['name'], 'Object1')

    def test_get_fewshot_object_types_empty(self):
        """Test getting few-shot object types when none exist"""
        self.mock_db.all.return_value = []

        resp = self.client.get('/api/fewshot/object-types')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data, [])

    def test_get_single_fewshot_object_type_success(self):
        """Test getting a single few-shot object type"""
        mock_object = MagicMock()
        self.mock_db.get.return_value = mock_object
        self.mock_fewshot_schema.dump.return_value = {
            'id': 'fs_123',
            'name': 'CustomObject',
            'description': 'A custom object'
        }

        resp = self.client.get('/api/fewshot/object-types/fs_123')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['id'], 'fs_123')
        self.assertEqual(data['name'], 'CustomObject')

    def test_get_single_fewshot_object_type_not_found(self):
        """Test getting a non-existent few-shot object type"""
        self.mock_db.get.return_value = None

        resp = self.client.get('/api/fewshot/object-types/nonexistent')
        
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertEqual(data['status'], 'fail')

    def test_delete_fewshot_object_type_success(self):
        """Test successful deletion of few-shot object type"""
        mock_object = MagicMock()
        self.mock_db.get.return_value = mock_object

        resp = self.client.delete('/api/fewshot/object-types/fs_123')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['message'], 'resource successfully deleted')
        self.mock_db.delete.assert_called_once_with(mock_object)

    def test_delete_fewshot_object_type_not_found(self):
        """Test deletion of non-existent few-shot object type"""
        self.mock_db.get.return_value = None

        resp = self.client.delete('/api/fewshot/object-types/nonexistent')
        
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertEqual(data['status'], 'fail')

    def test_get_fewshot_predictions_success(self):
        """Test getting few-shot predictions"""
        mock_predictions = [MagicMock(), MagicMock()]
        self.mock_db.all.return_value = mock_predictions
        self.mock_fewshot_pred_schema.dump.return_value = [
            {'id': 'pred_1', 'count': 2, 'confidence': 0.9},
            {'id': 'pred_2', 'count': 1, 'confidence': 0.8}
        ]

        resp = self.client.get('/api/fewshot/predictions')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['count'], 2)

    def test_get_fewshot_predictions_empty(self):
        """Test getting few-shot predictions when none exist"""
        self.mock_db.all.return_value = []

        resp = self.client.get('/api/fewshot/predictions')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data, [])


if __name__ == '__main__':
    unittest.main()

