#!/usr/bin/python3
"""Test helpers used across tests.

Provides utilities to reset the database to a clean state between tests and
to create a test Flask app with in-memory SQLite.
"""
from src import storage
from src.storage.base_model import Base

def reset_database():
    """Drop and recreate all tables to ensure a clean DB for tests.

    Uses the Engine inside src.storage.database. This function should be called
    in setUp() of test cases to guarantee isolation.
    """
    # engine object is stored as private attribute _engine on Engine instance
    engine = getattr(storage.database, "_engine", None)
    if engine is None:
        # Ensure session and engine created
        storage.database.reload()
        engine = getattr(storage.database, "_engine")
    # Drop & create all tables
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    # Reset the session to ensure no stale state
    storage.database.reload()
