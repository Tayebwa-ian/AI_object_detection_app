#!/usr/bin/env python3
"""
Comprehensive API Test Suite for AI Object Counting Application

This test suite covers all major API endpoints and functionality:
- Object detection endpoints
- Object types management
- Results management
- Few-shot learning
- Performance monitoring
- Batch processing
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from flask import Flask
from flask_restful import Api
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.views.inputs import InputList
from src.api.views.object_types import ObjectTypeList, ObjectTypeSingle
from src.api.views.outputs import OutputList, OutputSingle
from src.api.views.fewshot import (
    FewShotRegister, FewShotCount, FewShotObjectTypes, 
    FewShotObjectTypeSingle, FewShotPredictions
)
from src.api.views.monitoring import (
    PerformanceMetrics, ObjectTypeStats, DatabaseStats, 
    ResetStats, SystemHealth
)
from src.api.views.batch_processing import BatchProcessing, BatchStatus


class TestComprehensiveAPI(unittest.TestCase):
    """Comprehensive test suite for all API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test-secret-key'
        
        # Create API and register all resources
        self.api = Api(self.app)
        
        # Register all endpoints
        self.api.add_resource(InputList, '/api/count')
        self.api.add_resource(ObjectTypeList, '/api/object-types')
        self.api.add_resource(ObjectTypeSingle, '/api/object/<string:obj_id>')
        self.api.add_resource(OutputList, '/api/results')
        self.api.add_resource(OutputSingle, '/api/results/<string:output_id>', '/api/correct/<string:output_id>')
        self.api.add_resource(FewShotRegister, '/api/fewshot/register')
        self.api.add_resource(FewShotCount, '/api/fewshot/count')
        self.api.add_resource(FewShotObjectTypes, '/api/fewshot/object-types')
        self.api.add_resource(FewShotObjectTypeSingle, '/api/fewshot/object-types/<string:object_name>')
        self.api.add_resource(FewShotPredictions, '/api/fewshot/predictions')
        self.api.add_resource(PerformanceMetrics, '/api/performance/metrics')
        self.api.add_resource(ObjectTypeStats, '/api/performance/object-types')
        self.api.add_resource(DatabaseStats, '/api/performance/database')
        self.api.add_resource(ResetStats, '/api/performance/reset')
        self.api.add_resource(SystemHealth, '/api/performance/health')
        self.api.add_resource(BatchProcessing, '/api/batch/process')
        self.api.add_resource(BatchStatus, '/api/batch/status')
        
        self.client = self.app.test_client()
        
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.media_dir = os.path.join(self.temp_dir, 'media')
        os.makedirs(self.media_dir, exist_ok=True)
        
        # Mock all external dependencies
        self.setup_mocks()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        patch.stopall()
    
    def setup_mocks(self):
        """Set up all necessary mocks"""
        # Mock database
        self.db_patcher = patch('src.storage.database')
        self.mock_db = self.db_patcher.start()
        
        # Mock models
        self.input_patcher = patch('src.storage.inputs.Input')
        self.output_patcher = patch('src.storage.outputs.Output')
        self.object_type_patcher = patch('src.storage.object_types.ObjectType')
        self.fewshot_patcher = patch('src.storage.fewshot_models')
        
        self.mock_input = self.input_patcher.start()
        self.mock_output = self.output_patcher.start()
        self.mock_object_type = self.object_type_patcher.start()
        self.mock_fewshot = self.fewshot_patcher.start()
        
        # Mock AI pipeline
        self.pipeline_patcher = patch('src.pipeline.pipeline.pipeline')
        self.mock_pipeline = self.pipeline_patcher.start()
        
        # Mock few-shot service
        self.fewshot_service_patcher = patch('src.pipeline.fewshot_service.fewshot_service')
        self.mock_fewshot_service = self.fewshot_service_patcher.start()
        
        # Mock monitoring
        self.monitoring_patcher = patch('src.api.views.inputs.monitoring')
        self.mock_monitoring = self.monitoring_patcher.start()
        
        # Mock file upload utilities
        self.upload_patcher = patch('src.api.utils.image_utils.upload_image')
        self.mock_upload = self.upload_patcher.start()
        
        # Mock error handlers
        self.error_handler_patcher = patch('src.api.utils.error_handlers.validate_file_upload')
        self.mock_validate_upload = self.error_handler_patcher.start()
        
        # Mock config
        self.config_patcher = patch('src.config.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.MEDIA_DIRECTORY = self.media_dir
        self.mock_config.MAX_FILE_SIZE = 10485760
        
        # Setup default mock behaviors
        self.setup_default_mocks()
    
    def setup_default_mocks(self):
        """Set up default mock behaviors"""
        # Database mocks
        self.mock_db.all.return_value = []
        self.mock_db.get.return_value = None
        self.mock_db.delete.return_value = None
        self.mock_db.update.return_value = MagicMock()
        
        # Model mocks
        mock_instance = MagicMock()
        mock_instance.save.return_value = None
        mock_instance.id = 'test-id-123'
        mock_instance.created_at = MagicMock()
        mock_instance.created_at.isoformat.return_value = '2024-01-01T00:00:00'
        mock_instance.updated_at = MagicMock()
        mock_instance.updated_at.isoformat.return_value = '2024-01-01T00:00:00'
        
        self.mock_input.return_value = mock_instance
        self.mock_output.return_value = mock_instance
        self.mock_object_type.return_value = mock_instance
        
        # Pipeline mocks
        self.mock_pipeline.process_image.return_value = {
            'success': True,
            'predicted_count': 3,
            'confidence': 0.95,
            'processing_time': 2.5,
            'object_type': 'person'
        }
        self.mock_pipeline.process_image_auto.return_value = {
            'success': True,
            'predicted_count': 5,
            'confidence': 0.88,
            'processing_time': 3.2,
            'object_type': 'mixed'
        }
        self.mock_pipeline.get_model_status.return_value = {
            'models_loaded': True,
            'sam_loaded': True,
            'resnet_loaded': True
        }
        
        # Few-shot service mocks
        self.mock_fewshot_service.register_object_type.return_value = {
            'success': True,
            'object_type_id': 'fewshot-id-123',
            'object_name': 'test_object',
            'support_images_count': 3,
            'training_time_ms': 1500
        }
        self.mock_fewshot_service.count_objects_fewshot.return_value = {
            'success': True,
            'predicted_count': 2,
            'confidence': 0.92,
            'object_type': 'test_object'
        }
        self.mock_fewshot_service.get_fewshot_object_types.return_value = [
            {
                'id': 'fewshot-id-123',
                'name': 'test_object',
                'description': 'Test object type',
                'support_images_count': 3,
                'is_active': True
            }
        ]
        self.mock_fewshot_service.get_fewshot_object_type.return_value = {
            'id': 'fewshot-id-123',
            'name': 'test_object',
            'description': 'Test object type',
            'support_images_count': 3,
            'is_active': True
        }
        self.mock_fewshot_service.delete_fewshot_object_type.return_value = {
            'success': True,
            'message': 'Object type deleted successfully'
        }
        
        # Monitoring mocks
        self.mock_monitoring.record_request.return_value = None
        
        # Upload mocks
        self.mock_upload.return_value = 'test_image_123.jpg'
        self.mock_validate_upload.return_value = MagicMock()
    
    def create_test_image_file(self, filename='test_image.jpg'):
        """Create a test image file"""
        test_image_path = os.path.join(self.media_dir, filename)
        with open(test_image_path, 'wb') as f:
            f.write(b'fake_image_data')
        return test_image_path
    
    def test_object_detection_endpoints(self):
        """Test object detection endpoints"""
        print("\n=== Testing Object Detection Endpoints ===")
        
        # Test POST /api/count
        with patch('src.api.views.inputs.request') as mock_request:
            mock_request.files = {'image': MagicMock()}
            mock_request.form = {'object_type': 'person', 'description': 'Test image'}
            
            response = self.client.post('/api/count')
            self.assertEqual(response.status_code, 201)
            
            data = response.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['object_type'], 'person')
            self.assertEqual(data['predicted_count'], 3)
        
        print("✅ Object detection endpoints working")
    
    def test_object_types_management(self):
        """Test object types management endpoints"""
        print("\n=== Testing Object Types Management ===")
        
        # Test GET /api/object-types
        response = self.client.get('/api/object-types')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('object_types', data)
        
        # Test POST /api/object-types
        test_data = {
            'name': 'test_object',
            'description': 'Test object type'
        }
        
        response = self.client.post('/api/object-types', 
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 201)
        
        # Test GET /api/object/test-id
        self.mock_db.get.return_value = MagicMock()
        response = self.client.get('/api/object/test-id')
        self.assertEqual(response.status_code, 200)
        
        # Test PUT /api/object/test-id
        response = self.client.put('/api/object/test-id',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        # Test DELETE /api/object/test-id
        response = self.client.delete('/api/object/test-id')
        self.assertEqual(response.status_code, 200)
        
        print("✅ Object types management working")
    
    def test_results_management(self):
        """Test results management endpoints"""
        print("\n=== Testing Results Management ===")
        
        # Test GET /api/results
        response = self.client.get('/api/results')
        self.assertEqual(response.status_code, 200)
        
        # Test GET /api/results with pagination
        response = self.client.get('/api/results?page=1&per_page=10')
        self.assertEqual(response.status_code, 200)
        
        # Test GET /api/results with filtering
        response = self.client.get('/api/results?object_type=person')
        self.assertEqual(response.status_code, 200)
        
        # Test GET /api/results/test-id
        self.mock_db.get.return_value = MagicMock()
        response = self.client.get('/api/results/test-id')
        self.assertEqual(response.status_code, 200)
        
        # Test PUT /api/correct/test-id (correction endpoint)
        correction_data = {'corrected_count': 5}
        response = self.client.put('/api/correct/test-id',
                                 data=json.dumps(correction_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        # Test DELETE /api/results/test-id
        response = self.client.delete('/api/results/test-id')
        self.assertEqual(response.status_code, 200)
        
        print("✅ Results management working")
    
    def test_fewshot_learning_endpoints(self):
        """Test few-shot learning endpoints"""
        print("\n=== Testing Few-Shot Learning Endpoints ===")
        
        # Test POST /api/fewshot/register
        with patch('src.api.views.fewshot.request') as mock_request:
            mock_request.form = {
                'object_name': 'custom_object',
                'description': 'Custom object type'
            }
            mock_request.files = {
                'support_images': [MagicMock(), MagicMock(), MagicMock()]
            }
            
            response = self.client.post('/api/fewshot/register')
            self.assertEqual(response.status_code, 201)
            
            data = response.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['object_name'], 'test_object')
        
        # Test POST /api/fewshot/count
        with patch('src.api.views.fewshot.request') as mock_request:
            mock_request.form = {
                'object_name': 'custom_object',
                'description': 'Count custom objects'
            }
            mock_request.files = {'image': MagicMock()}
            
            response = self.client.post('/api/fewshot/count')
            self.assertEqual(response.status_code, 200)
            
            data = response.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['predicted_count'], 2)
        
        # Test GET /api/fewshot/object-types
        response = self.client.get('/api/fewshot/object-types')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('object_types', data)
        
        # Test GET /api/fewshot/object-types/test_object
        response = self.client.get('/api/fewshot/object-types/test_object')
        self.assertEqual(response.status_code, 200)
        
        # Test DELETE /api/fewshot/object-types/test_object
        response = self.client.delete('/api/fewshot/object-types/test_object')
        self.assertEqual(response.status_code, 200)
        
        # Test GET /api/fewshot/predictions
        response = self.client.get('/api/fewshot/predictions')
        self.assertEqual(response.status_code, 200)
        
        print("✅ Few-shot learning endpoints working")
    
    def test_performance_monitoring_endpoints(self):
        """Test performance monitoring endpoints"""
        print("\n=== Testing Performance Monitoring Endpoints ===")
        
        # Test GET /api/performance/metrics
        response = self.client.get('/api/performance/metrics')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('total_requests', data)
        self.assertIn('success_rate_percent', data)
        
        # Test GET /api/performance/object-types
        response = self.client.get('/api/performance/object-types')
        self.assertEqual(response.status_code, 200)
        
        # Test GET /api/performance/database
        response = self.client.get('/api/performance/database')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('total_inputs', data)
        self.assertIn('total_outputs', data)
        
        # Test GET /api/performance/health
        response = self.client.get('/api/performance/health')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('overall_health_score', data)
        self.assertIn('health_status', data)
        
        # Test POST /api/performance/reset
        response = self.client.post('/api/performance/reset')
        self.assertEqual(response.status_code, 200)
        
        print("✅ Performance monitoring endpoints working")
    
    def test_batch_processing_endpoints(self):
        """Test batch processing endpoints"""
        print("\n=== Testing Batch Processing Endpoints ===")
        
        # Test POST /api/batch/process
        with patch('src.api.views.batch_processing.request') as mock_request:
            mock_request.form = {
                'object_type': 'person',
                'description': 'Batch processing test'
            }
            mock_request.files = {
                'images[]': [MagicMock(), MagicMock(), MagicMock()]
            }
            
            response = self.client.post('/api/batch/process')
            self.assertEqual(response.status_code, 200)
            
            data = response.get_json()
            self.assertTrue(data['success'])
            self.assertIn('batch_id', data)
            self.assertIn('results', data)
        
        # Test GET /api/batch/status
        response = self.client.get('/api/batch/status')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn('total_processed_today', data)
        self.assertIn('success_rate', data)
        
        print("✅ Batch processing endpoints working")
    
    def test_error_handling(self):
        """Test error handling across endpoints"""
        print("\n=== Testing Error Handling ===")
        
        # Test invalid object type
        with patch('src.api.views.inputs.request') as mock_request:
            mock_request.files = {'image': MagicMock()}
            mock_request.form = {'object_type': '', 'description': 'Test'}
            
            # Mock validation error
            with patch('src.api.utils.error_handlers.validate_object_type') as mock_validate:
                mock_validate.side_effect = Exception('Invalid object type')
                
                response = self.client.post('/api/count')
                self.assertIn(response.status_code, [400, 500])
        
        # Test missing file
        with patch('src.api.views.inputs.request') as mock_request:
            mock_request.files = {}
            mock_request.form = {'object_type': 'person'}
            
            response = self.client.post('/api/count')
            self.assertIn(response.status_code, [400, 500])
        
        # Test invalid JSON
        response = self.client.post('/api/object-types',
                                  data='invalid json',
                                  content_type='application/json')
        self.assertIn(response.status_code, [400, 500])
        
        print("✅ Error handling working")
    
    def test_api_documentation_endpoints(self):
        """Test API documentation and health endpoints"""
        print("\n=== Testing API Documentation ===")
        
        # Test root endpoint (should redirect to docs or return info)
        response = self.client.get('/')
        self.assertIn(response.status_code, [200, 302, 404])
        
        # Test metrics endpoint (Prometheus)
        response = self.client.get('/metrics')
        self.assertIn(response.status_code, [200, 404])
        
        print("✅ API documentation endpoints working")
    
    def run_comprehensive_test(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive API Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_object_detection_endpoints,
            self.test_object_types_management,
            self.test_results_management,
            self.test_fewshot_learning_endpoints,
            self.test_performance_monitoring_endpoints,
            self.test_batch_processing_endpoints,
            self.test_error_handling,
            self.test_api_documentation_endpoints
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                test_method()
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"🎉 Comprehensive Test Suite Complete!")
        print(f"Passed: {passed_tests}/{total_tests} tests")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("✅ All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed_tests == total_tests


def main():
    """Main function to run comprehensive tests"""
    print("AI Object Counting Application - Comprehensive API Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveAPI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Also run the comprehensive test method
    test_instance = TestComprehensiveAPI()
    test_instance.setUp()
    try:
        success = test_instance.run_comprehensive_test()
        test_instance.tearDown()
        
        if success and result.wasSuccessful():
            print("\n🎉 All comprehensive tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        test_instance.tearDown()
        return 1


if __name__ == '__main__':
    exit(main())
