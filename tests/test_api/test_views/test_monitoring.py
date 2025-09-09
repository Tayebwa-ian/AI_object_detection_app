#!/usr/bin/env python3
"""
Test Monitoring API Views
"""
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_restful import Api

from src.api.views.monitoring import (
    PerformanceMetrics, ObjectTypeStats, DatabaseStats, 
    ResetStats, SystemHealth
)


class TestMonitoringViews(unittest.TestCase):
    """Test Monitoring API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        app = Flask(__name__)
        api = Api(app)
        api.add_resource(PerformanceMetrics, '/api/performance/metrics')
        api.add_resource(ObjectTypeStats, '/api/performance/object-types')
        api.add_resource(DatabaseStats, '/api/performance/database')
        api.add_resource(ResetStats, '/api/performance/reset')
        api.add_resource(SystemHealth, '/api/performance/health')

        self.app = app
        self.client = app.test_client()

        # Patch dependencies
        self.metrics_patcher = patch('src.api.views.monitoring.metrics')
        self.db_patcher = patch('src.api.views.monitoring.database')

        self.mock_metrics = self.metrics_patcher.start()
        self.mock_db = self.db_patcher.start()

    def tearDown(self):
        """Clean up patches"""
        patch.stopall()

    def test_performance_metrics_success(self):
        """Test getting performance metrics"""
        # Mock metrics data
        self.mock_metrics.get_performance_metrics.return_value = {
            'total_requests': 1000,
            'successful_requests': 950,
            'failed_requests': 50,
            'average_response_time': 1.5,
            'requests_per_minute': 60,
            'uptime': 3600
        }

        resp = self.client.get('/api/performance/metrics')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('total_requests', data)
        self.assertIn('successful_requests', data)
        self.assertIn('failed_requests', data)
        self.assertEqual(data['total_requests'], 1000)
        self.assertEqual(data['successful_requests'], 950)

    def test_performance_metrics_error(self):
        """Test performance metrics with error"""
        self.mock_metrics.get_performance_metrics.side_effect = Exception('Metrics error')

        resp = self.client.get('/api/performance/metrics')
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Metrics error', data['message'])

    def test_object_type_stats_success(self):
        """Test getting object type statistics"""
        # Mock object type stats
        self.mock_metrics.get_object_type_stats.return_value = {
            'car': {'count': 100, 'accuracy': 0.95},
            'person': {'count': 50, 'accuracy': 0.90},
            'bicycle': {'count': 25, 'accuracy': 0.85}
        }

        resp = self.client.get('/api/performance/object-types')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('car', data)
        self.assertIn('person', data)
        self.assertIn('bicycle', data)
        self.assertEqual(data['car']['count'], 100)
        self.assertEqual(data['car']['accuracy'], 0.95)

    def test_object_type_stats_error(self):
        """Test object type stats with error"""
        self.mock_metrics.get_object_type_stats.side_effect = Exception('Stats error')

        resp = self.client.get('/api/performance/object-types')
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Stats error', data['message'])

    def test_database_stats_success(self):
        """Test getting database statistics"""
        # Mock database stats
        self.mock_db.get_stats.return_value = {
            'total_inputs': 500,
            'total_outputs': 450,
            'total_object_types': 10,
            'database_size': '50MB',
            'last_backup': '2024-01-01T00:00:00Z'
        }

        resp = self.client.get('/api/performance/database')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('total_inputs', data)
        self.assertIn('total_outputs', data)
        self.assertIn('total_object_types', data)
        self.assertEqual(data['total_inputs'], 500)
        self.assertEqual(data['total_outputs'], 450)

    def test_database_stats_error(self):
        """Test database stats with error"""
        self.mock_db.get_stats.side_effect = Exception('Database error')

        resp = self.client.get('/api/performance/database')
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Database error', data['message'])

    def test_reset_stats_success(self):
        """Test resetting statistics"""
        self.mock_metrics.reset_stats.return_value = True

        resp = self.client.post('/api/performance/reset')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'Statistics reset successfully')
        self.mock_metrics.reset_stats.assert_called_once()

    def test_reset_stats_error(self):
        """Test reset stats with error"""
        self.mock_metrics.reset_stats.side_effect = Exception('Reset error')

        resp = self.client.post('/api/performance/reset')
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Reset error', data['message'])

    def test_system_health_success(self):
        """Test system health check"""
        # Mock system health data
        self.mock_metrics.get_system_health.return_value = {
            'status': 'healthy',
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'disk_usage': 23.1,
            'database_connection': True,
            'ai_models_loaded': True,
            'last_check': '2024-01-01T00:00:00Z'
        }

        resp = self.client.get('/api/performance/health')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('status', data)
        self.assertIn('cpu_usage', data)
        self.assertIn('memory_usage', data)
        self.assertIn('database_connection', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['database_connection'])

    def test_system_health_unhealthy(self):
        """Test system health check when unhealthy"""
        # Mock unhealthy system
        self.mock_metrics.get_system_health.return_value = {
            'status': 'unhealthy',
            'cpu_usage': 95.0,
            'memory_usage': 98.5,
            'disk_usage': 95.2,
            'database_connection': False,
            'ai_models_loaded': False,
            'last_check': '2024-01-01T00:00:00Z'
        }

        resp = self.client.get('/api/performance/health')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'unhealthy')
        self.assertFalse(data['database_connection'])
        self.assertFalse(data['ai_models_loaded'])

    def test_system_health_error(self):
        """Test system health check with error"""
        self.mock_metrics.get_system_health.side_effect = Exception('Health check error')

        resp = self.client.get('/api/performance/health')
        
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Health check error', data['message'])

    def test_performance_metrics_empty_data(self):
        """Test performance metrics with empty data"""
        self.mock_metrics.get_performance_metrics.return_value = {}

        resp = self.client.get('/api/performance/metrics')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data, {})

    def test_object_type_stats_empty_data(self):
        """Test object type stats with empty data"""
        self.mock_metrics.get_object_type_stats.return_value = {}

        resp = self.client.get('/api/performance/object-types')
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data, {})


if __name__ == '__main__':
    unittest.main()
