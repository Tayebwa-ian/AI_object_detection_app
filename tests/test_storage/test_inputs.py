#!/usr/bin/python3
"""Unit tests for Input model."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.inputs import Input


class TestInputModel(unittest.TestCase):
    """Tests for Input model fields and relationships."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_input_and_flags(self):
        inp = Input(image_path="tests/data/sample.jpg", prompt="Count cars", is_few_shot=True)
        self.sess.add(inp)
        self.sess.commit()

        fetched = self.sess.query(Input).filter_by(image_path="tests/data/sample.jpg").one_or_none()
        self.assertIsNotNone(fetched)
        self.assertTrue(fetched.is_few_shot)
        self.assertFalse(fetched.is_zero_shot)

    def test_violation_count_non_negative(self):
        inp = Input(image_path="img.png", violation_count=3)
        self.sess.add(inp)
        self.sess.commit()
        fetched = self.sess.query(Input).get(inp.id)
        self.assertGreaterEqual(fetched.violation_count, 0)


if __name__ == "__main__":
    unittest.main()
