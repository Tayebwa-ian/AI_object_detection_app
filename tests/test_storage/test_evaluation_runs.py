#!/usr/bin/python3
"""Unit tests for EvaluationRun model and association with ModelLabel / InferencePeriod."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.evaluation_runs import EvaluationRun
from src.storage.ai_models import AIModel
from src.storage.models_labels import ModelLabel
from src.storage.inference_periods import InferencePeriod
from src.storage.labels import Label
from src.storage.inputs import Input


class TestEvaluationRunModel(unittest.TestCase):
    """Create a run and attach model metrics & latencies."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_run_associations(self):
        m = AIModel(name="eval-model")
        l = Label(name="eval-label")
        i = Input(image_path="eval-img.jpg")
        self.sess.add_all([m, l, i])
        self.sess.commit()

        run = EvaluationRun(ai_model_id=m.id, run_type="test", metadata='{"note":"smoke"}')
        self.sess.add(run)
        self.sess.flush()

        ml = ModelLabel(ai_model_id=m.id, label_id=l.id, accuracy=0.5, run_id=run.id)
        ip = InferencePeriod(ai_model_id=m.id, input_id=i.id, value=0.2, run_id=run.id)
        self.sess.add_all([ml, ip])
        self.sess.commit()

        fetched_run = self.sess.query(EvaluationRun).get(run.id)
        self.assertEqual(len(fetched_run.model_labels), 1)
        self.assertEqual(len(fetched_run.inference_periods), 1)


if __name__ == "__main__":
    unittest.main()
