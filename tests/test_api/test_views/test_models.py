#!/usr/bin/python3
"""Tests for AIModel endpoints including training & testing flows."""

import unittest
import os
import json
from tests.test_helpers import reset_database
from src.app import create_app
from src import storage
from src.storage.labels import Label
from src.storage.inputs import Input

os.environ["OBJ_DETECT_ENV"] = "test"


class TestModelsViews(unittest.TestCase):
    """Integration tests for model CRUD + train/test endpoints."""

    @classmethod
    def setUpClass(cls):
        reset_database()
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_and_get_model(self):
        payload = {"name": "new-model", "description": "testing"}
        r = self.client.post("/api/v1/models", json=payload)
        self.assertEqual(r.status_code, 201)
        js = r.get_json()
        model_id = js["id"]

        # fetch
        gr = self.client.get(f"/api/v1/models/{model_id}")
        self.assertEqual(gr.status_code, 200)
        gjs = gr.get_json()
        self.assertEqual(gjs["name"], "new-model")

    def test_train_model_simulated(self):
        payload = {"name": "trainable"}
        r = self.client.post("/api/v1/models", json=payload)
        self.assertEqual(r.status_code, 201)
        model_id = r.get_json()["id"]

        tr_payload = {"dataset": "d", "epochs": 2, "params": {"lr": 0.001}}
        tr = self.client.post(f"/api/v1/models/{model_id}/train", json=tr_payload)
        self.assertEqual(tr.status_code, 201)
        trj = tr.get_json()
        self.assertIn("run_id", trj)

    def test_test_model_store_results(self):
        # create model, label and input
        model = self.sess.query(storage.database.session.bind) if False else None
        r1 = self.client.post("/api/v1/models", json={"name": "for-test"})
        model_id = r1.get_json()["id"]

        lab = Label(name="dog")
        inp = Input(image_path="t1.jpg")
        self.sess.add_all([lab, inp])
        self.sess.commit()

        payload = {
            "dataset": "testset",
            "results": [{"label_id": lab.id, "accuracy": 0.8, "precision": 0.7, "recall": 0.75, "f1_score": 0.72}],
            "latencies": [{"input_id": inp.id, "value": 0.12}],
            "metadata": {"note": "ci"}
        }
        r = self.client.post(f"/api/v1/models/{model_id}/test", json=payload)
        self.assertEqual(r.status_code, 201)
        js = r.get_json()
        self.assertEqual(js["model_labels_inserted"], 1)
        self.assertEqual(js["latencies_inserted"], 1)


if __name__ == "__main__":
    unittest.main()
