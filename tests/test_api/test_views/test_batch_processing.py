#!/usr/bin/python3
"""
Unit tests for BatchProcessing API endpoints
"""
import io
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from werkzeug.datastructures import FileStorage

# Import the module with corrected path
from src.api.views.batch_processing import BatchProcessing, BatchStatus

class TestBatchProcessing(unittest.TestCase):
    """Tests for BatchProcessing endpoint"""

    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.bp = BatchProcessing()
        self.app.add_url_rule(
            "/api/batch/process",
            view_func=BatchProcessing.as_view("batch_process")
        )

    @patch("src.api.views.batch_processing.BatchProcessing._process_single_image")
    @patch("src.api.views.batch_processing.validate_object_type", return_value="car")
    @patch("src.api.views.batch_processing.monitoring")
    def test_post_batch_success(self, mock_monitoring, mock_validate, mock_process_single):
        """POST /batch/process should process images successfully"""
        # Mock _process_single_image to always return success
        mock_process_single.return_value = {
            'image_name': 'test.jpg',
            'success': True,
            'result_id': 'output-1',
            'object_type': 'car',
            'predicted_count': 2,
            'confidence': 0.95,
            'processing_time': 0.1,
            'created_at': '2025-01-01T00:00:00'
        }

        # Create a fake file upload
        data = {
            "object_type": "car",
            "images[]": (io.BytesIO(b"fake image data"), "test.jpg")
        }

        with self.app.test_request_context(
            "/api/batch/process",
            method="POST",
            content_type="multipart/form-data",
            data=data
        ):
            response = self.bp.post()
            status_code = response.status_code if hasattr(response, "status_code") else response[1]
            json_data = response.get_json() if hasattr(response, "get_json") else response[0]

        self.assertEqual(status_code, 200)
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["total_images"], 1)
        self.assertEqual(json_data["successful_images"], 1)
        self.assertEqual(len(json_data["results"]), 1)
        self.assertTrue(json_data["results"][0]["success"])

    @patch("src.api.views.batch_processing.database")
    @patch("src.api.views.batch_processing.monitoring")
    def test_get_batch_status_success(self, mock_monitoring, mock_db):
        """GET /batch/status should return batch statistics"""
        # Mock database.all to return empty list
        mock_db.all.return_value = []

        # Mock monitoring.get_metrics
        mock_monitoring.get_metrics.return_value = {
            'average_processing_time': 0.12,
            'success_rate_percent': 100,
            'total_requests': 10,
            'uptime_seconds': 3600
        }

        bp_status = BatchStatus()
        with self.app.test_request_context("/api/batch/status", method="GET"):
            response = bp_status.get()
            status_code = response[1]
            json_data = response[0]

        self.assertEqual(status_code, 200)
        self.assertIn("total_processed_today", json_data)
        self.assertIn("average_processing_time", json_data)
        self.assertIn("success_rate", json_data)
        self.assertIn("total_requests", json_data)
        self.assertIn("system_uptime", json_data)
        self.assertIn("last_updated", json_data)

    @patch("src.api.views.batch_processing.create_error_response")
    def test_post_batch_no_images(self, mock_create_error):
        """POST /batch/process with no images should return validation error"""
        mock_create_error.return_value = ("error_response", 400)

        bp = BatchProcessing()
        with self.app.test_request_context("/api/batch/process", method="POST", data={}):
            response = bp.post()

        self.assertEqual(response, ("error_response", 400))

    @patch("src.api.views.batch_processing.BatchProcessing._process_single_image")
    @patch("src.api.views.batch_processing.validate_object_type", return_value="car")
    def test_post_batch_too_many_images(self, mock_validate, mock_process_single):
        """POST /batch/process with >10 images should return validation error"""
        mock_process_single.return_value = {
            'image_name': 'test.jpg',
            'success': True,
            'result_id': 'output-1',
            'object_type': 'car',
            'predicted_count': 2,
            'confidence': 0.95,
            'processing_time': 0.1,
            'created_at': '2025-01-01T00:00:00'
        }

        # Create 11 fake files
        data = {
            "object_type": "car",
            "images[]": [(io.BytesIO(b"data"), f"file{i}.jpg") for i in range(11)]
        }

        bp = BatchProcessing()
        with self.app.test_request_context(
            "/api/batch/process",
            method="POST",
            content_type="multipart/form-data",
            data=data
        ):
            response = bp.post()

        # Assert Flask Response object and status code
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("Too many images", json_data["error"]["message"])


if __name__ == "__main__":
    unittest.main()
