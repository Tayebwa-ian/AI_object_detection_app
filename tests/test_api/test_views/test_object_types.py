import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, json
from src.api.views.object_types import ObjectTypeList, ObjectTypeSingle


class TestObjectTypeList(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.app.add_url_rule('/object-types', view_func=ObjectTypeList.as_view('object_types_list'))

    @patch("src.api.views.object_types.database")
    def test_get_object_types_existing(self, mock_db):
        """GET /object-types should return list if objects exist"""
        mock_obj = MagicMock()
        mock_db.all.return_value = [mock_obj]
        with patch("src.api.views.object_types.objs_schema.dump", return_value=[{"name": "car"}]):
            response = self.client.get("/object-types")
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("object_types", data)
        self.assertEqual(data["object_types"][0]["name"], "car")

    @patch("src.api.views.object_types.database")
    @patch("src.api.views.object_types.ObjectType")
    def test_get_object_types_creates_defaults(self, mock_obj_type, mock_db):
        """GET /object-types should create defaults if none exist"""
        # First call: no objects in DB
        mock_db.all.side_effect = [[], [MagicMock()]]
        mock_db.get.return_value = None
        mock_instance = MagicMock()
        mock_obj_type.return_value = mock_instance

        with patch("src.api.views.object_types.objs_schema.dump", return_value=[{"name": "person"}]):
            response = self.client.get("/object-types")
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("object_types", data)
        self.assertEqual(data["object_types"][0]["name"], "person")
        mock_instance.save.assert_called()  # ensure defaults were saved

    @patch("src.api.views.object_types.ObjectType")
    def test_post_object_type_success(self, mock_obj_type):
        """POST /object-types should create new object type"""
        mock_instance = MagicMock()
        mock_obj_type.return_value = mock_instance
        mock_instance.id = 1

        with patch("src.api.views.object_types.obj_schema.load", return_value={"name": "tree", "description": "Tall plant"}), \
             patch("src.api.views.object_types.obj_schema.dump", return_value={"id": 1, "name": "tree"}):
            response = self.client.post(
                "/object-types",
                data=json.dumps({"name": "tree", "description": "Tall plant"}),
                content_type="application/json"
            )
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(data["name"], "tree")

    def test_post_object_type_validation_error(self):
        """POST /object-types with invalid payload should return 403"""
        response = self.client.post(
            "/object-types",
            data=json.dumps({"bad_field": "oops"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 403)


class TestObjectTypeSingle(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.app.add_url_rule('/object-types/<int:obj_id>', view_func=ObjectTypeSingle.as_view('object_type_single'))

    @patch("src.api.views.object_types.database")
    def test_get_single_object_type(self, mock_db):
        mock_obj = MagicMock()
        mock_db.get.return_value = mock_obj
        with patch("src.api.views.object_types.obj_schema.dump", return_value={"id": 1, "name": "car"}):
            response = self.client.get("/object-types/1")
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["name"], "car")

    @patch("src.api.views.object_types.database")
    def test_delete_object_type(self, mock_db):
        mock_obj = MagicMock()
        mock_db.get.return_value = mock_obj
        response = self.client.delete("/object-types/1")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["message"], "resource successfully deleted")

    @patch("src.api.views.object_types.database")
    def test_put_object_type_success(self, mock_db):
        mock_obj = MagicMock()
        mock_db.update.return_value = mock_obj
        with patch("src.api.views.object_types.obj_schema.load", return_value={"name": "dog"}), \
             patch("src.api.views.object_types.obj_schema.dump", return_value={"id": 1, "name": "dog"}):
            response = self.client.put(
                "/object-types/1",
                data=json.dumps({"name": "dog", "description": "Canine"}),
                content_type="application/json"
            )
            data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["name"], "dog")

    def test_put_object_type_validation_error(self):
        response = self.client.put(
            "/object-types/1",
            data=json.dumps({"bad_field": "oops"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
