#!/usr/bin/python3
"""Unit tests for InferencePeriod model (latency records)."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.inference_periods import InferencePeriod
from src.storage.ai_models import AIModel
from src.storage.inputs import Input


class TestInferencePeriodModel(unittest.TestCase):
    """Test latency records and relationships."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_inference_period(self):
        m = AIModel(name="lat-model")
        i = Input(image_path="img-lat.png")
        self.sess.add_all([m, i])
        self.sess.commit()

        ip = InferencePeriod(ai_model_id=m.id, input_id=i.id, value=0.123)
        self.sess.add(ip)
        self.sess.commit()

        fetched = self.sess.query(InferencePeriod).get(ip.id)
        self.assertIsNotNone(fetched)
        self.assertAlmostEqual(fetched.value, 0.123)
        self.assertEqual(fetched.ai_model.id, m.id)
        self.assertEqual(fetched.input.id, i.id)


if __name__ == "__main__":
    unittest.main()
