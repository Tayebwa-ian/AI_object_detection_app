# tests/test_outputs.py
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_restful import Api
from marshmallow import ValidationError

from src.api.views.outputs import OutputList, OutputSingle


class TestOutputViews(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        api = Api(app)
        api.add_resource(OutputList, '/api/outputs')
        api.add_resource(OutputSingle, '/api/outputs/<string:output_id>')

        self.app = app
        self.client = app.test_client()

        # Patch module-level dependencies
        self.db_patcher = patch('src.api.views.outputs.database')
        self.output_schema_patcher = patch('src.api.views.outputs.output_schema')
        self.outputs_schema_patcher = patch('src.api.views.outputs.outputs_schema')
        self.Output_patcher = patch('src.api.views.outputs.Output')

        self.mock_db = self.db_patcher.start()
        self.mock_output_schema = self.output_schema_patcher.start()
        self.mock_outputs_schema = self.outputs_schema_patcher.start()
        self.mock_Output = self.Output_patcher.start()

        self.mock_output_schema.load = MagicMock()
        self.mock_output_schema.dump = MagicMock()
        self.mock_outputs_schema.dump = MagicMock()

    def tearDown(self):
        patch.stopall()

    def test_get_all_outputs_empty_returns_400(self):
        self.mock_db.all.return_value = []
        resp = self.client.get('/api/outputs')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('message', data)

    def test_get_all_outputs_success_returns_200(self):
        fake_model = MagicMock()
        self.mock_db.all.return_value = [fake_model]
        self.mock_outputs_schema.dump.return_value = [{'id': 'o1', 'predicted_count': 3}]

        resp = self.client.get('/api/outputs')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), [{'id': 'o1', 'predicted_count': 3}])

    def test_post_output_success_returns_201(self):
        payload = {
            'predicted_count': 5,
            'pred_confidence': 0.9,
            'object_type_id': 'otype-1',
            'input_id': 'input-1'
        }
        # note: view calls load twice in original code; returning same result is fine
        self.mock_output_schema.load.return_value = payload
        fake_instance = MagicMock()
        fake_instance.save = MagicMock()
        self.mock_Output.return_value = fake_instance
        self.mock_output_schema.dump.return_value = {'id': 'o1', **payload}

        resp = self.client.post('/api/outputs', json=payload)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json(), {'id': 'o1', **payload})
        fake_instance.save.assert_called_once()

    def test_post_output_validation_error_returns_403(self):
        payload = {}
        self.mock_output_schema.load.side_effect = ValidationError({'predicted_count': ['required']})

        resp = self.client.post('/api/outputs', json=payload)
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'fail')

    def test_get_single_output_success(self):
        fake_instance = MagicMock()
        self.mock_db.get.return_value = fake_instance
        self.mock_output_schema.dump.return_value = {'id': 'o1', 'predicted_count': 3}

        resp = self.client.get('/api/outputs/o1')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {'id': 'o1', 'predicted_count': 3})

    def test_delete_output_calls_delete_and_returns_200(self):
        fake_instance = MagicMock()
        self.mock_db.get.return_value = fake_instance
        resp = self.client.delete('/api/outputs/o1')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('message'), 'resource successfully deleted')
        self.mock_db.delete.assert_called_once_with(fake_instance)

    def test_put_output_success_returns_200(self):
        payload = {'predicted_count': 6, 'pred_confidence': 0.88}
        self.mock_output_schema.load.return_value = payload
        updated_instance = MagicMock()
        self.mock_db.update.return_value = updated_instance
        self.mock_output_schema.dump.return_value = {'id': 'o1', **payload}

        resp = self.client.put('/api/outputs/o1', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {'id': 'o1', **payload})
        self.mock_db.update.assert_called_once()

    def test_put_output_validation_error_returns_403(self):
        payload = {}
        self.mock_output_schema.load.side_effect = ValidationError({'predicted_count': ['required']})
        resp = self.client.put('/api/outputs/o1', json=payload)
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'fail')


if __name__ == '__main__':
    unittest.main()
