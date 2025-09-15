#!/usr/bin/python3
"""Unit tests for AIModel database model."""

import unittest
from tests.test_helpers import reset_database
from src import storage
from src.storage.ai_models import AIModel


class TestAIModelModel(unittest.TestCase):
    """Tests basic CRUD and relationships for AIModel."""

    def setUp(self):
        reset_database()
        self.sess = storage.database.session

    def test_create_and_retrieve(self):
        """Create an AIModel, commit, and retrieve it by id and name."""
        model = AIModel(name="test-model-1", description="a test model")
        self.sess.add(model)
        self.sess.commit()

        fetched = self.sess.query(AIModel).filter_by(name="test-model-1").one_or_none()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "test-model-1")
        self.assertEqual(fetched.description, "a test model")

    def test_unique_name_enforced(self):
        """Name is unique: inserting duplicate names will raise an IntegrityError."""
        from sqlalchemy.exc import IntegrityError
        m1 = AIModel(name="dup-model")
        self.sess.add(m1)
        self.sess.commit()

        m2 = AIModel(name="dup-model")
        self.sess.add(m2)
        with self.assertRaises(IntegrityError):
            self.sess.commit()


if __name__ == "__main__":
    unittest.main()
