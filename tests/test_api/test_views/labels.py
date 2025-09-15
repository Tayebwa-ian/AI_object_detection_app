#!/usr/bin/python3
"""Tests for label endpoints (safety & CRUD)."""

import unittest
import os
from tests.test_helpers import reset_database
from src.app import create_app
from src import storage

os.environ["OBJ_DETECT_ENV"] = "test"


class TestLabelsViews(unittest.TestCase):
    """Integration tests covering label creation and safety checks."""

    @classmethod
    def setUpClass(cls):
        reset_database()
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_label_reject_military(self):
        payload = {"name": "military vehicle", "description": "not allowed"}
        r = self.client.post("/api/v1/labels", json=payload)
        self.assertEqual(r.status_code, 422)
        js = r.get_json()
        self.assertIn("errors", js) or self.assertIn("message", js)

    def test_create_label_success(self):
        payload = {"name": "bicycle", "description": "two-wheeled vehicle"}
        r = self.client.post("/api/v1/labels", json=payload)
        self.assertEqual(r.status_code, 201)
        js = r.get_json()
        self.assertEqual(js["name"], "bicycle")
        # ensure we can fetch it
        lbl_id = js["id"]
        get_r = self.client.get(f"/api/v1/labels/{lbl_id}")
        self.assertEqual(get_r.status_code, 200)
        get_js = get_r.get_json()
        self.assertEqual(get_js["name"], "bicycle")


if __name__ == "__main__":
    unittest.main()
