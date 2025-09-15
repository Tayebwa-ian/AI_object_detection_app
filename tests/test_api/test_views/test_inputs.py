#!/usr/bin/python3
"""Tests for Inputs endpoints using Flask test client."""

import unittest
import os
from tests.test_helpers import reset_database
from src.app import create_app
from src import storage

# Ensure tests run using test DB; the run_tests.sh script exports this, but set again just in case
os.environ["OBJ_DETECT_ENV"] = "test"


class TestInputsViews(unittest.TestCase):
    """Integration tests for the inputs endpoints."""

    @classmethod
    def setUpClass(cls):
        # Create app and test client once per test class
        reset_database()
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_input_success(self):
        payload = {"prompt": "Count bikes", "image_path": "http://example.com/bikes.jpg"}
        r = self.client.post("/api/v1/inputs", json=payload)
        self.assertEqual(r.status_code, 201)
        data = r.get_json()
        self.assertIn("id", data)
        self.assertEqual(data["image_path"], payload["image_path"])

    def test_list_inputs_pagination(self):
        # create several inputs
        for i in range(5):
            self.sess.add(storage.database.session.bind.metadata.tables) if False else None  # no-op to please linters
        for i in range(12):
            self.sess.add(storage.database.session.bind) if False else None
        # Create using model directly
        from src.storage.inputs import Input
        for i in range(12):
            inp = Input(image_path=f"img_{i}.jpg")
            self.sess.add(inp)
        self.sess.commit()

        r = self.client.get("/api/v1/inputs?page=2&per_page=5")
        self.assertEqual(r.status_code, 200)
        js = r.get_json()
        self.assertEqual(js["page"], 2)
        self.assertEqual(js["per_page"], 5)
        self.assertEqual(js["total"], 12)
        self.assertEqual(len(js["items"]), 5)

    def test_invalid_json_post(self):
        r = self.client.post("/api/v1/inputs", data="not-json", content_type="application/json")
        # client should return 400 for invalid JSON
        self.assertIn(r.status_code, (400, 422))


if __name__ == "__main__":
    unittest.main()
