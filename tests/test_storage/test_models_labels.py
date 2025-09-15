#!/usr/bin/python3
"""Unit tests for ModelLabel aggregated metrics model."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.models_labels import ModelLabel
from src.storage.ai_models import AIModel
from src.storage.labels import Label


class TestModelLabelModel(unittest.TestCase):
    """Tests ModelLabel creation and relationships."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_model_label(self):
        m = AIModel(name="m1")
        l = Label(name="cat")
        self.sess.add_all([m, l])
        self.sess.commit()

        ml = ModelLabel(ai_model_id=m.id, label_id=l.id, accuracy=0.9, precision=0.8, recall=0.85, f1_score=0.825)
        self.sess.add(ml)
        self.sess.commit()

        fetched = self.sess.query(ModelLabel).filter_by(ai_model_id=m.id, label_id=l.id).one_or_none()
        self.assertIsNotNone(fetched)
        self.assertAlmostEqual(fetched.accuracy, 0.9)
        self.assertEqual(fetched.ai_model.id, m.id)
        self.assertEqual(fetched.label.id, l.id)


if __name__ == "__main__":
    unittest.main()
