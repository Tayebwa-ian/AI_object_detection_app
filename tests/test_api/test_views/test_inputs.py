# tests/test_inputs.py
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_restful import Api
from marshmallow import ValidationError

# Import the resource classes (adjust import path if your project differs)
from src.api.views.inputs import InputList, InputSingle


class TestInputViews(unittest.TestCase):
    def setUp(self):
        # Create a Flask app and register the resources
        app = Flask(__name__)
        api = Api(app)
        api.add_resource(InputList, '/api/inputs')
        api.add_resource(InputSingle, '/api/inputs/<string:input_id>')

        self.app = app
        self.client = app.test_client()

        # Patch database, schemas, and model in the module where they're used
        self.db_patcher = patch('src.api.views.inputs.database')
        self.input_schema_patcher = patch('src.api.views.inputs.input_schema')
        self.inputs_schema_patcher = patch('src.api.views.inputs.inputs_schema')
        self.Input_patcher = patch('src.api.views.inputs.Input')

        self.mock_db = self.db_patcher.start()
        self.mock_input_schema = self.input_schema_patcher.start()
        self.mock_inputs_schema = self.inputs_schema_patcher.start()
        self.mock_Input = self.Input_patcher.start()

        # Make sure dump/load are MagicMocks
        self.mock_input_schema.load = MagicMock()
        self.mock_input_schema.dump = MagicMock()
        self.mock_inputs_schema.dump = MagicMock()

    def tearDown(self):
        patch.stopall()

    def test_get_all_inputs_empty_returns_400(self):
        self.mock_db.all.return_value = []
        resp = self.client.get('/api/inputs')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('message', data)
        self.assertIn('could not fetch data', data['message'])

    def test_get_all_inputs_success_returns_200(self):
        fake_model = MagicMock()
        self.mock_db.all.return_value = [fake_model]
        self.mock_inputs_schema.dump.return_value = [{'id': '1', 'description': 'x'}]

        resp = self.client.get('/api/inputs')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), [{'id': '1', 'description': 'x'}])

    def test_post_input_success_returns_201(self):
        payload = {'description': 'a', 'image_path': '/img.jpg'}
        # schema.load returns validated data
        self.mock_input_schema.load.return_value = payload
        # Input() constructor returns an instance whose save method will be called
        fake_instance = MagicMock()
        fake_instance.save = MagicMock()
        self.mock_Input.return_value = fake_instance
        # dump returns serialized object
        self.mock_input_schema.dump.return_value = {'id': 'uuid', **payload}

        resp = self.client.post('/api/inputs', json=payload)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json(), {'id': 'uuid', **payload})
        fake_instance.save.assert_called_once()

    def test_post_input_validation_error_returns_403(self):
        payload = {'description': ''}
        self.mock_input_schema.load.side_effect = ValidationError({'description': ['required']})

        resp = self.client.post('/api/inputs', json=payload)
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'fail')
        self.assertIn('description', data.get('message'))

    def test_get_single_input_success(self):
        fake_instance = MagicMock()
        self.mock_db.get.return_value = fake_instance
        self.mock_input_schema.dump.return_value = {'id': 'uuid', 'description': 'a'}

        resp = self.client.get('/api/inputs/uuid')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {'id': 'uuid', 'description': 'a'})

    def test_delete_input_calls_delete_and_returns_200(self):
        fake_instance = MagicMock()
        self.mock_db.get.return_value = fake_instance
        resp = self.client.delete('/api/inputs/uuid')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('message'), 'resource successfully deleted')
        self.mock_db.delete.assert_called_once_with(fake_instance)

    def test_put_input_success_returns_200(self):
        payload = {'description': 'updated', 'image_path': '/new.png'}
        self.mock_input_schema.load.return_value = payload
        updated_instance = MagicMock()
        self.mock_db.update.return_value = updated_instance
        self.mock_input_schema.dump.return_value = {'id': 'uuid', **payload}

        resp = self.client.put('/api/inputs/uuid', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {'id': 'uuid', **payload})
        self.mock_db.update.assert_called_once()

    def test_put_input_validation_error_returns_403(self):
        payload = {}
        self.mock_input_schema.load.side_effect = ValidationError({'description': ['missing']})
        resp = self.client.put('/api/inputs/uuid', json=payload)
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'fail')


if __name__ == '__main__':
    unittest.main()
