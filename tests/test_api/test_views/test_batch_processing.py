#!/usr/bin/env python3
"""
Test Batch Processing API Views
"""
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_restful import Api
from marshmallow import ValidationError

from src.api.views.batch_processing import BatchProcessing, BatchStatus


class TestBatchProcessingViews(unittest.TestCase):
    """Test Batch Processing API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        app = Flask(__name__)
        api = Api(app)
        api.add_resource(BatchProcessing, '/api/batch/process')
        api.add_resource(BatchStatus, '/api/batch/status')

        self.app = app
        self.client = app.test_client()

        # Patch dependencies
        self.pipeline_patcher = patch('src.api.views.batch_processing.pipeline')
        self.db_patcher = patch('src.api.views.batch_processing.database')
        self.input_schema_patcher = patch('src.api.views.batch_processing.input_schema')
        self.output_schema_patcher = patch('src.api.views.batch_processing.output_schema')

        self.mock_pipeline = self.pipeline_patcher.start()
        self.mock_db = self.db_patcher.start()
        self.mock_input_schema = self.input_schema_patcher.start()
        self.mock_output_schema = self.output_schema_patcher.start()

        # Setup mock methods
        self.mock_input_schema.load = MagicMock()
        self.mock_output_schema.dump = MagicMock()

    def tearDown(self):
        """Clean up patches"""
        patch.stopall()

    def test_batch_process_success(self):
        """Test successful batch processing"""
        payload = {
            'images': [
                {'image_path': '/test/image1.jpg', 'description': 'Test image 1'},
                {'image_path': '/test/image2.jpg', 'description': 'Test image 2'}
            ]
        }
        
        # Mock pipeline processing
        self.mock_pipeline.process_image.return_value = {
            'objects': [{'type': 'car', 'confidence': 0.9}],
            'counts': {'car': 2}
        }
        
        # Mock database operations
        mock_input = MagicMock()
        mock_input.id = 'input_123'
        self.mock_db.save.return_value = mock_input
        
        mock_output = MagicMock()
        self.mock_output_schema.dump.return_value = {
            'id': 'output_123',
            'predicted_count': 2,
            'pred_confidence': 0.9
        }

        resp = self.client.post('/api/batch/process', json=payload)
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('results', data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['results']), 2)

    def test_batch_process_validation_error(self):
        """Test batch processing with validation error"""
        payload = {'images': []}  # Empty images list
        self.mock_input_schema.load.side_effect = ValidationError({'images': ['required']})

        resp = self.client.post('/api/batch/process', json=payload)
        
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data['status'], 'fail')

    def test_batch_process_pipeline_error(self):
        """Test batch processing with pipeline error"""
        payload = {
            'images': [
                {'image_path': '/test/image1.jpg', 'description': 'Test image 1'}
            ]
        }
        
        # Mock pipeline to raise an error
        self.mock_pipeline.process_image.side_effect = Exception('Processing failed')

        resp = self.client.post('/api/batch/process', json=payload)
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Processing failed', data['message'])

    def test_batch_process_partial_success(self):
        """Test batch processing with partial success"""
        payload = {
            'images': [
                {'image_path': '/test/image1.jpg', 'description': 'Test image 1'},
                {'image_path': '/test/image2.jpg', 'description': 'Test image 2'}
            ]
        }
        
        # Mock pipeline to succeed for first image, fail for second
        def mock_process_side_effect(image_path, *args, **kwargs):
            if 'image1' in image_path:
                return {
                    'objects': [{'type': 'car', 'confidence': 0.9}],
                    'counts': {'car': 1}
                }
            else:
                raise Exception('Processing failed for image2')
        
        self.mock_pipeline.process_image.side_effect = mock_process_side_effect
        
        # Mock database operations
        mock_input = MagicMock()
        mock_input.id = 'input_123'
        self.mock_db.save.return_value = mock_input

        resp = self.client.post('/api/batch/process', json=payload)
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('results', data)
        self.assertIn('errors', data)
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(len(data['errors']), 1)

    def test_batch_status_success(self):
        """Test getting batch processing status"""
        # Mock database to return batch statistics
        self.mock_db.get_batch_stats.return_value = {
            'total_batches': 10,
            'successful_batches': 8,
            'failed_batches': 2,
            'total_images_processed': 50,
            'average_processing_time': 2.5
        }

        resp = self.client.get('/api/batch/status')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('total_batches', data)
        self.assertIn('successful_batches', data)
        self.assertIn('failed_batches', data)
        self.assertEqual(data['total_batches'], 10)
        self.assertEqual(data['successful_batches'], 8)

    def test_batch_status_database_error(self):
        """Test batch status with database error"""
        self.mock_db.get_batch_stats.side_effect = Exception('Database error')

        resp = self.client.get('/api/batch/status')
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Database error', data['message'])

    def test_batch_process_large_batch(self):
        """Test batch processing with large number of images"""
        # Create a large batch of images
        images = []
        for i in range(100):
            images.append({
                'image_path': f'/test/image{i}.jpg',
                'description': f'Test image {i}'
            })
        
        payload = {'images': images}
        
        # Mock pipeline processing
        self.mock_pipeline.process_image.return_value = {
            'objects': [{'type': 'car', 'confidence': 0.9}],
            'counts': {'car': 1}
        }
        
        # Mock database operations
        mock_input = MagicMock()
        mock_input.id = 'input_123'
        self.mock_db.save.return_value = mock_input

        resp = self.client.post('/api/batch/process', json=payload)
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 100)

    def test_batch_process_no_images(self):
        """Test batch processing with no images"""
        payload = {'images': []}

        resp = self.client.post('/api/batch/process', json=payload)
        
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data['status'], 'fail')
        self.assertIn('No images provided', data['message'])


if __name__ == '__main__':
    unittest.main()
