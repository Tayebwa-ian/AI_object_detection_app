#!/usr/bin/python3
"""Unit tests for monitoring endpoints.

Covers:
 - /metrics (Prometheus scrape)
 - /api/v1/metrics/summary (JSON aggregated top labels)
 - /api/v1/metrics/query (flexible query by approach, model, label, input, metric)

These tests use the in-memory SQLite database (OBJ_DETECT_ENV=test). Each test
resets the DB to a clean state to ensure isolation.
"""
import os
import unittest
from tests.test_helpers import reset_database
from src.app import create_app
from src import storage

# Ensure the Engine runs in test mode (in-memory sqlite) for these tests
os.environ["OBJ_DETECT_ENV"] = "test"
os.environ["OBJ_DETECT_USE_SQLITE"] = "1"
# Optional: ensure sqlite file setting does not interfere
os.environ["OBJ_DETECT_SQLITE_FILE"] = ":memory:"


class TestMonitoringEndpoints(unittest.TestCase):
    """Integration tests for monitoring endpoints."""

    @classmethod
    def setUpClass(cls):
        """Create Flask test app once for this test class."""
        reset_database()
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()

    def setUp(self):
        """Reset DB and obtain a fresh session for every test."""
        reset_database()
        self.sess = storage.database.session

    def test_metrics_scrape_and_content_type(self):
        """Prometheus scrape should return text/plain metrics payload (200)."""
        # Create minimal data so compute_metrics has something to aggregate.
        from src.storage.ai_models import AIModel
        from src.storage.labels import Label
        from src.storage.inputs import Input
        from src.storage.outputs import Output

        model = AIModel(name="scrape-model")
        label = Label(name="scrape-label")
        inp = Input(image_path="scrape.jpg")
        self.sess.add_all([model, label, inp])
        self.sess.commit()

        out = Output(input_id=inp.id, label_id=label.id, ai_model_id=model.id,
                     predicted_count=1, confidence=0.55)
        self.sess.add(out)
        self.sess.commit()

        # Hit scrape endpoint
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        # Content type for Prometheus text format
        self.assertIn("text/plain", r.headers.get("Content-Type", "") or r.headers.get("content-type", ""))
        # Basic sanity check: generated payload contains HELP/TYPE lines or our metric names
        body = r.data.decode("utf-8")
        self.assertTrue(body.strip() != "")  # not empty
        # Our gauge names should appear (or at least HELP lines)
        self.assertTrue(("# HELP" in body) or ("object_model_avg_confidence" in body) or ("object_model_metric" in body))

    def test_metrics_summary_returns_top_labels(self):
        """Metrics summary should return mapping model_name -> list of labels sorted by avg_confidence."""
        from src.storage.ai_models import AIModel
        from src.storage.labels import Label
        from src.storage.inputs import Input
        from src.storage.outputs import Output

        m = AIModel(name="sum-model")
        l1 = Label(name="label-a")
        l2 = Label(name="label-b")
        i1 = Input(image_path="sum1.jpg", is_few_shot=True)
        i2 = Input(image_path="sum2.jpg", is_zero_shot=True)
        self.sess.add_all([m, l1, l2, i1, i2])
        self.sess.commit()

        # outputs: label-a has higher avg_conf than label-b
        o1 = Output(input_id=i1.id, label_id=l1.id, ai_model_id=m.id, predicted_count=1, confidence=0.9)
        o2 = Output(input_id=i2.id, label_id=l1.id, ai_model_id=m.id, predicted_count=1, confidence=0.8)
        o3 = Output(input_id=i1.id, label_id=l2.id, ai_model_id=m.id, predicted_count=1, confidence=0.2)
        self.sess.add_all([o1, o2, o3])
        self.sess.commit()

        r = self.client.get("/api/v1/metrics/summary?top_n_labels=5")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        # Should contain the model name as a key and a list of labels
        self.assertIn(m.name, data)
        labels_list = data[m.name]
        self.assertIsInstance(labels_list, list)
        # label-a should come first with higher avg_confidence
        self.assertGreaterEqual(labels_list[0]["avg_confidence"], labels_list[-1]["avg_confidence"])

    def test_metrics_query_confidence_by_approach(self):
        """Query confidence aggregated by approach (few_shot / zero_shot / none)."""
        from src.storage.ai_models import AIModel
        from src.storage.labels import Label
        from src.storage.inputs import Input
        from src.storage.outputs import Output

        m = AIModel(name="mq-model")
        l = Label(name="mq-label")
        i1 = Input(image_path="qa_few.jpg", is_few_shot=True)
        i2 = Input(image_path="qa_zero.jpg", is_zero_shot=True)
        i3 = Input(image_path="qa_none.jpg", is_few_shot=False, is_zero_shot=False)
        self.sess.add_all([m, l, i1, i2, i3])
        self.sess.commit()

        # Create outputs with different confidences
        o1 = Output(input_id=i1.id, label_id=l.id, ai_model_id=m.id, predicted_count=1, confidence=0.7)  # few_shot
        o2 = Output(input_id=i2.id, label_id=l.id, ai_model_id=m.id, predicted_count=1, confidence=0.2)  # zero_shot
        o3 = Output(input_id=i3.id, label_id=l.id, ai_model_id=m.id, predicted_count=1, confidence=0.5)  # none
        self.sess.add_all([o1, o2, o3])
        self.sess.commit()

        # Query few_shot
        r_few = self.client.get("/api/v1/metrics/query?metric=confidence&agg=avg&group_by=model,label&approach=few_shot")
        self.assertEqual(r_few.status_code, 200)
        jf = r_few.get_json()
        self.assertEqual(jf["metric"], "confidence")
        self.assertEqual(jf["agg"], "avg")
        # results should include at least one entry and value ~0.7
        self.assertTrue(isinstance(jf["results"], list) and len(jf["results"]) >= 1)
        found = False
        for row in jf["results"]:
            # check that returned rows reference our model id or name
            if row.get("ai_model_name") == m.name or row.get("ai_model_id") == m.id:
                found = True
                # value should equal 0.7 (within tolerance)
                self.assertAlmostEqual(row["value"], 0.7, places=3)
        self.assertTrue(found)

        # Query zero_shot
        r_zero = self.client.get("/api/v1/metrics/query?metric=confidence&agg=avg&group_by=model,label&approach=zero_shot")
        self.assertEqual(r_zero.status_code, 200)
        jz = r_zero.get_json()
        self.assertTrue(len(jz["results"]) >= 1)
        # find model entry and value ~0.2
        found_zero = any((row.get("value") == 0.2 or abs(row.get("value", 0) - 0.2) < 1e-6) for row in jz["results"])
        self.assertTrue(found_zero)

        # Query none
        r_none = self.client.get("/api/v1/metrics/query?metric=confidence&agg=avg&group_by=model,label&approach=none")
        self.assertEqual(r_none.status_code, 200)
        jn = r_none.get_json()
        # Should reflect the "none" approach and value ~0.5
        found_none = any(abs(row.get("value", 0) - 0.5) < 1e-6 for row in jn["results"])
        self.assertTrue(found_none)

    def test_metrics_query_latency_and_grouping(self):
        """Query latency metric aggregated by model and by model+input groupings."""
        from src.storage.ai_models import AIModel
        from src.storage.inputs import Input
        from src.storage.inference_periods import InferencePeriod

        m1 = AIModel(name="lat-model-1")
        m2 = AIModel(name="lat-model-2")
        i1 = Input(image_path="lat-1.jpg")
        i2 = Input(image_path="lat-2.jpg")
        self.sess.add_all([m1, m2, i1, i2])
        self.sess.commit()

        # Create latency records
        ip1 = InferencePeriod(ai_model_id=m1.id, input_id=i1.id, value=0.1)
        ip2 = InferencePeriod(ai_model_id=m1.id, input_id=i2.id, value=0.3)
        ip3 = InferencePeriod(ai_model_id=m2.id, input_id=i1.id, value=0.2)
        self.sess.add_all([ip1, ip2, ip3])
        self.sess.commit()

        # Query avg latency grouped by model
        r_model = self.client.get("/api/v1/metrics/query?metric=latency&agg=avg&group_by=model")
        self.assertEqual(r_model.status_code, 200)
        jm = r_model.get_json()
        self.assertEqual(jm["metric"], "latency")
        # results list should include both models
        models_returned = {row.get("ai_model_name") or row.get("ai_model_id") for row in jm["results"]}
        self.assertTrue(m1.name in models_returned or m2.name in models_returned)

        # Query avg latency grouped by model,input
        r_mi = self.client.get("/api/v1/metrics/query?metric=latency&agg=avg&group_by=model,input")
        self.assertEqual(r_mi.status_code, 200)
        jmi = r_mi.get_json()
        # Each entry should contain ai_model_id and input_id and a numeric value
        for row in jmi["results"]:
            self.assertIn("ai_model_id", row)
            self.assertIn("input_id", row)
            self.assertIn("value", row)

        # Query latency grouped by label should return 400 (unsupported)
        r_bad = self.client.get("/api/v1/metrics/query?metric=latency&agg=avg&group_by=label")
        self.assertEqual(r_bad.status_code, 400)

    def test_metrics_query_modellevel_metrics(self):
        """Query accuracy/precision/recall/f1 from ModelLabel table."""
        from src.storage.ai_models import AIModel
        from src.storage.labels import Label
        from src.storage.models_labels import ModelLabel

        m = AIModel(name="mmodel")
        l = Label(name="mlabel")
        self.sess.add_all([m, l])
        self.sess.commit()

        # Create a few ModelLabel rows (simulate multiple runs)
        ml1 = ModelLabel(ai_model_id=m.id, label_id=l.id, accuracy=0.9, precision=0.85, recall=0.8, f1_score=0.825)
        ml2 = ModelLabel(ai_model_id=m.id, label_id=l.id, accuracy=0.95, precision=0.9, recall=0.85, f1_score=0.875)
        self.sess.add_all([ml1, ml2])
        self.sess.commit()

        r = self.client.get("/api/v1/metrics/query?metric=accuracy&agg=avg&group_by=model,label")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j["metric"], "accuracy")
        # There should be at least one result and average should be (0.9+0.95)/2 = 0.925
        found_avg = False
        for row in j["results"]:
            # Match model id or name
            if row.get("ai_model_name") == m.name or row.get("ai_model_id") == m.id:
                # value approximate
                self.assertAlmostEqual(row["value"], 0.925, places=3)
                found_avg = True
        self.assertTrue(found_avg)

    def test_metrics_query_invalid_params(self):
        """Invalid metric or aggregation should produce HTTP 400."""
        r1 = self.client.get("/api/v1/metrics/query?metric=not_a_metric&agg=avg")
        self.assertEqual(r1.status_code, 400)
        r2 = self.client.get("/api/v1/metrics/query?metric=confidence&agg=notagg")
        self.assertEqual(r2.status_code, 400)
        # Invalid group_by part should also return 400
        r3 = self.client.get("/api/v1/metrics/query?metric=confidence&agg=avg&group_by=unknown")
        self.assertEqual(r3.status_code, 400)


if __name__ == "__main__":
    unittest.main()
