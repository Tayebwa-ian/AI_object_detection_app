#!/usr/bin/python3
"""Unit tests for Output model and its relationships."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.outputs import Output
from src.storage.inputs import Input
from src.storage.labels import Label
from src.storage.ai_models import AIModel


class TestOutputModel(unittest.TestCase):
    """Create sample Input, Label and AIModel and link an Output to them."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_output_relationships(self):
        inp = Input(image_path="img1.jpg")
        lbl = Label(name="truck")
        model = AIModel(name="model-x")
        self.sess.add_all([inp, lbl, model])
        self.sess.commit()

        out = Output(input_id=inp.id, label_id=lbl.id, ai_model_id=model.id, predicted_count=2, confidence=0.75)
        self.sess.add(out)
        self.sess.commit()

        # verify relationships
        fetched_out = self.sess.query(Output).get(out.id)
        self.assertEqual(fetched_out.input.id, inp.id)
        self.assertEqual(fetched_out.label.id, lbl.id)
        self.assertEqual(fetched_out.ai_model.id, model.id)
        self.assertAlmostEqual(fetched_out.confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
