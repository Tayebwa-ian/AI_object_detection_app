# tests/test_labels.py
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_restful import Api
from marshmallow import ValidationError

from src.api.views.labels import LabelList, LabelSingle


class TestLabelViews(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        api = Api(app)
        api.add_resource(LabelList, '/api/labels')
        api.add_resource(LabelSingle, '/api/labels/<string:label_id>')

        self.app = app
        self.client = app.test_client()

        # Patch module-level dependencies
        self.db_patcher = patch('src.api.views.labels.database')
        self.label_schema_patcher = patch('src.api.views.labels.label_schema')
        self.labels_schema_patcher = patch('src.api.views.labels.labels_schema')
        self.labelectType_patcher = patch('src.api.views.labels.Label')

        self.mock_db = self.db_patcher.start()
        self.mock_obj_schema = self.obj_schema_patcher.start()
        self.mock_objs_schema = self.objs_schema_patcher.start()
        self.mock_ObjectType = self.ObjectType_patcher.start()

        self.mock_obj_schema.load = MagicMock()
        self.mock_obj_schema.dump = MagicMock()
        self.mock_objs_schema.dump = MagicMock()

    def tearDown(self):
        patch.stopall()

    def test_get_all_labels_empty_returns_400(self):
        self.mock_db.all.return_value = []
        resp = self.client.get('/api/labels')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('message', data)

    def test_get_all_labels_success_returns_200(self):
        fake_model = MagicMock()
        self.mock_db.all.return_value = [fake_model]
        self.mock_objs_schema.dump.return_value = [{'id': 't1', 'name': 'Car'}]

        resp = self.client.get('/api/labels')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), [{'id': 't1', 'name': 'Car'}])

    def test_post_labels_success_returns_201(self):
        payload = {'name': 'Tree', 'description': 'Tall plant'}
        self.mock_obj_schema.load.return_value = payload
        fake_instance = MagicMock()
        fake_instance.save = MagicMock()
        self.mock_ObjectType.return_value = fake_instance
        self.mock_obj_schema.dump.return_value = {'id': 'ot1', **payload}

        resp = self.client.post('/api/labels', json=payload)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json(), {'id': 'ot1', **payload})
        fake_instance.save.assert_called_once()

    def test_post_labels_validation_error_returns_403(self):
        payload = {}
        self.mock_obj_schema.load.side_effect = ValidationError({'name': ['required']})
        resp = self.client.post('/api/labels', json=payload)
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'fail')

    def test_get_single_labels_success(self):
        fake_instance = MagicMock()
        self.mock_db.get.return_value = fake_instance
        self.mock_obj_schema.dump.return_value = {'id': 'ot1', 'name': 'Car'}

        resp = self.client.get('/api/labels/ot1')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {'id': 'ot1', 'name': 'Car'})

    def test_delete_labels_calls_delete_and_returns_200(self):
        fake_instance = MagicMock()
        self.mock_db.get.return_value = fake_instance
        resp = self.client.delete('/api/labels/ot1')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('message'), 'resource successfully deleted')
        self.mock_db.delete.assert_called_once_with(fake_instance)

    def test_put_labels_success_returns_200(self):
        payload = {'name': 'Building', 'description': 'Structure'}
        self.mock_obj_schema.load.return_value = payload
        updated_instance = MagicMock()
        self.mock_db.update.return_value = updated_instance
        self.mock_obj_schema.dump.return_value = {'id': 'ot1', **payload}

        resp = self.client.put('/api/labels/ot1', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {'id': 'ot1', **payload})
        self.mock_db.update.assert_called_once()

    def test_put_labels_validation_error_returns_403(self):
        payload = {}
        self.mock_obj_schema.load.side_effect = ValidationError({'name': ['required']})
        resp = self.client.put('/api/labels/ot1', json=payload)
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'fail')


if __name__ == '__main__':
    unittest.main()
