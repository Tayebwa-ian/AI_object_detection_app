#!/usr/bin/python3
"""Unit tests for Label model."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.labels import Label


class TestLabelModel(unittest.TestCase):
    """Tests for label model operations."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_label(self):
        label = Label(name="car", description="a road vehicle")
        self.sess.add(label)
        self.sess.commit()
        fetched = self.sess.query(Label).filter_by(name="car").one_or_none()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.description, "a road vehicle")

    def test_unique_name_constraint(self):
        from sqlalchemy.exc import IntegrityError
        l1 = Label(name="person")
        self.sess.add(l1)
        self.sess.commit()
        l2 = Label(name="person")
        self.sess.add(l2)
        with self.assertRaises(IntegrityError):
            self.sess.commit()


if __name__ == "__main__":
    unittest.main()
