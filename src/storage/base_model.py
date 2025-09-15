#!/usr/bin/python3
"""Base Model - common attributes for ORM models.

This defines `Base` (SQLAlchemy declarative base) and `BaseModel`.
BaseModel is a mixin that provides id, created_at, updated_at and small helpers.
Note: we avoid importing storage at module import-time to prevent circular imports;
storage is imported lazily inside methods.
"""
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime
from uuid import uuid4
from typing import Any, Dict

Base = declarative_base()


class BaseModel:
    """Common fields and helpers for all models.

    Fields:
        id (str): UUID primary key
        created_at (datetime): creation timestamp
        updated_at (datetime): last update timestamp
    """

    id = Column(String(60), primary_key=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now(), nullable=False)
    updated_at = Column(DateTime, default=datetime.now(), nullable=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the BaseModel. Accepts keyword overrides for fields."""
        # only set attributes if not provided by SQLAlchemy defaults or kwargs
        if not getattr(self, "id", None):
            self.id = str(uuid4())
        now = datetime.now()
        if not getattr(self, "created_at", None):
            self.created_at = now
        if not getattr(self, "updated_at", None):
            self.updated_at = now

        # allow setting other fields via kwargs (helpful in tests)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self) -> None:
        """Persist the instance using the configured storage session.

        This lazily imports the storage module to avoid circular imports.
        """
        from src import storage  # lazy import to avoid circular dependency
        self.updated_at = datetime.now()
        storage.database.new(self)
        storage.database.save()

    def delete(self) -> None:
        """Delete the instance using the configured storage session."""
        from src import storage  # lazy import
        storage.database.delete(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict representation excluding SQLAlchemy internal state."""
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_sa_instance_state"):
                continue
            d[k] = v
        return d

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"[{self.__class__.__name__}] ({self.id}) {self.to_dict()}"
