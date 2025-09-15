#!/usr/bin/python3
"""Tests for Outputs endpoints."""

import unittest
import os
from tests.test_helpers import reset_database
from src.app import create_app
from src import storage
from src.storage.inputs import Input
from src.storage.labels import Label
from src.storage.ai_models import AIModel

os.environ["OBJ_DETECT_ENV"] = "test"


class TestOutputsViews(unittest.TestCase):
    """Integration tests for output creation and listing."""

    @classmethod
    def setUpClass(cls):
        reset_database()
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()

    def setUp(self):
        reset_database()
        self.sess = storage.database.session
        # create supporting rows
        self.input = Input(image_path="out-img.jpg")
        self.label = Label(name="car")
        self.model = AIModel(name="test-model")
        self.sess.add_all([self.input, self.label, self.model])
        self.sess.commit()

    def test_create_output_success(self):
        payload = {
            "input_id": self.input.id,
            "label_id": self.label.id,
            "ai_model_id": self.model.id,
            "predicted_count": 3,
            "confidence": 0.88
        }
        r = self.client.post("/api/v1/outputs", json=payload)
        self.assertEqual(r.status_code, 201)
        js = r.get_json()
        self.assertEqual(js["predicted_count"], 3)
        self.assertAlmostEqual(js["confidence"], 0.88)

    def test_list_outputs_filter_by_model(self):
        # create two outputs: one for this model, one for a different model
        from src.storage.outputs import Output
        other = AIModel(name="other-model")
        self.sess.add(other)
        self.sess.commit()
        o1 = Output(input_id=self.input.id, label_id=self.label.id, ai_model_id=self.model.id, predicted_count=1)
        o2 = Output(input_id=self.input.id, label_id=self.label.id, ai_model_id=other.id, predicted_count=2)
        self.sess.add_all([o1, o2])
        self.sess.commit()

        r = self.client.get(f"/api/v1/outputs?ai_model_id={self.model.id}")
        self.assertEqual(r.status_code, 200)
        js = r.get_json()
        self.assertTrue(all(item["ai_model_id"] == self.model.id for item in js["items"]))


if __name__ == "__main__":
    unittest.main()
