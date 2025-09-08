import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, json
from src.api.views.outputs import OutputList, OutputSingle


class TestOutputList(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.app.add_url_rule('/outputs', view_func=OutputList.as_view('output_list'))

    @patch("src.api.views.outputs.database")
    def test_get_outputs_success(self, mock_db):
        """GET /outputs returns list of enhanced outputs"""
        mock_output = MagicMock()
        mock_output.id = 1
        mock_output.object_type_id = "ot1"
        mock_output.input_id = "in1"
        mock_output.predicted_count = 5
        mock_output.corrected_count = 4
        mock_output.pred_confidence = 0.95
        mock_output.created_at = MagicMock()
        mock_output.updated_at = MagicMock()
        mock_output.created_at.isoformat.return_value = "2025-01-01T00:00:00"
        mock_output.updated_at.isoformat.return_value = "2025-01-01T01:00:00"

        mock_db.all.return_value = [mock_output]

        mock_object_type = MagicMock()
        mock_object_type.name = "car"
        mock_input = MagicMock()
        mock_input.image_path = "/path/to/image.jpg"
        mock_db.get.side_effect = [mock_object_type, mock_input]

        response = self.client.get("/outputs")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["object_type"], "car")
        self.assertEqual(data[0]["image_path"], "/path/to/image.jpg")

    @patch("src.api.views.outputs.database")
    def test_get_outputs_with_filter(self, mock_db):
        """GET /outputs supports object_type filter"""
        mock_output = MagicMock()
        mock_output.id = 1
        mock_output.object_type_id = "ot1"
        mock_output.input_id = "in1"
        mock_output.predicted_count = 5
        mock_output.corrected_count = 4
        mock_output.pred_confidence = 0.95
        mock_output.created_at = MagicMock()
        mock_output.updated_at = MagicMock()
        mock_output.created_at.isoformat.return_value = "2025-01-01T00:00:00"
        mock_output.updated_at.isoformat.return_value = "2025-01-01T01:00:00"

        mock_db.all.return_value = [mock_output]

        mock_object_type = MagicMock()
        mock_object_type.name = "dog"
        mock_input = MagicMock()
        mock_input.image_path = "/img.jpg"
        mock_db.get.side_effect = [mock_object_type, mock_input]

        response = self.client.get("/outputs?object_type=dog")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["object_type"], "dog")

    @patch("src.api.views.outputs.Output")
    def test_post_output_success(self, mock_output_cls):
        """POST /outputs creates new output"""
        mock_instance = MagicMock()
        mock_output_cls.return_value = mock_instance
        mock_instance.id = 1

        with patch("src.api.views.outputs.output_schema.load", return_value={"predicted_count": 5}), \
             patch("src.api.views.outputs.output_schema.dump", return_value={"id": 1, "predicted_count": 5}):
            response = self.client.post(
                "/outputs",
                data=json.dumps({"predicted_count": 5, "pred_confidence": 0.9, "object_type_id": "ot1", "input_id": "in1"}),
                content_type="application/json"
            )
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(data["id"], 1)

    def test_post_output_validation_error(self):
        """POST /outputs with invalid data returns 403"""
        response = self.client.post(
            "/outputs",
            data=json.dumps({"invalid": "field"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 403)


class TestOutputSingle(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.app.add_url_rule('/outputs/<int:output_id>', view_func=OutputSingle.as_view('output_single'))

    @patch("src.api.views.outputs.database")
    def test_get_single_output_success(self, mock_db):
        mock_output = MagicMock()
        mock_output.id = 1
        mock_output.object_type_id = "ot1"
        mock_output.input_id = "in1"

        mock_object_type = MagicMock()
        mock_object_type.name = "car"

        mock_input = MagicMock()
        mock_input.image_path = "/img.jpg"

        mock_db.get.side_effect = [mock_output, mock_object_type, mock_input]

        with patch("src.api.views.outputs.output_schema.dump", return_value={"id": 1, "predicted_count": 5}):
            response = self.client.get("/outputs/1")
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["id"], 1)
        self.assertEqual(data["object_type"], "car")
        self.assertEqual(data["image_path"], "/img.jpg")

    @patch("src.api.views.outputs.database")
    def test_delete_output(self, mock_db):
        mock_output = MagicMock()
        mock_db.get.return_value = mock_output
        response = self.client.delete("/outputs/1")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["message"], "resource successfully deleted")

    @patch("src.api.views.outputs.database")
    def test_put_output_success(self, mock_db):
        mock_output = MagicMock()
        mock_output.id = 1
        mock_output.predicted_count = 5
        mock_output.updated_at = MagicMock()
        mock_output.updated_at.isoformat.return_value = "2025-01-01T02:00:00"
        mock_db.get.return_value = mock_output

        response = self.client.put(
            "/outputs/1",
            data=json.dumps({"corrected_count": 3}),
            content_type="application/json"
        )
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["corrected_count"], 3)
        self.assertTrue(data["success"])

    @patch("src.api.views.outputs.database")
    def test_put_output_missing_corrected_count(self, mock_db):
        mock_output = MagicMock()
        mock_db.get.return_value = mock_output

        response = self.client.put(
            "/outputs/1",
            data=json.dumps({}),
            content_type="application/json"
        )
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
