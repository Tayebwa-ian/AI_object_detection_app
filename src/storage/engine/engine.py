#!/usr/bin/python3
"""Engine - database connection & session manager.

This module centralizes creation of the SQLAlchemy engine and session.
It supports:
 - specifying a full DATABASE_URL environment var (recommended)
 - or MySQL config variables for production
 - or using SQLite for development/testing (first-class option)

Environment variables:
 - DATABASE_URL : full SQLAlchemy URL (e.g. mysql+mysqldb://user:pwd@host/db or sqlite:////path/to/file)
 - OBJ_DETECT_ENV : 'test'|'development'|'production' (affects defaults)
 - OBJ_DETECT_USE_SQLITE : '1' to force sqlite (dev)
 - OBJ_DETECT_MYSQL_USER, OBJ_DETECT_MYSQL_PWD, OBJ_DETECT_MYSQL_HOST, OBJ_DETECT_MYSQL_DB
"""
from os import getenv
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from src.storage.base_model import Base
from datetime import datetime
from typing import Optional


class Engine:
    """Manage a SQLAlchemy engine and a scoped session.

    Use database.session to access the SQLAlchemy Session object.
    """

    def __init__(self) -> None:
        """Set up engine and session factory. Actual Session instance is created via reload()."""
        self._engine = None
        self._session = None

        # Priority: explicit DATABASE_URL > use sqlite if requested > MySQL config
        database_url = getenv("DATABASE_URL")
        env = getenv("OBJ_DETECT_ENV", "development")
        use_sqlite_flag = getenv("OBJ_DETECT_USE_SQLITE", "0") == "1"

        if database_url:
            exec_db = database_url
        elif env == "test":
            # In-memory sqlite for tests (clean slate, fast)
            exec_db = "sqlite:///:memory:"
        elif use_sqlite_flag:
            # development sqlite file path (default)
            sqlite_file = getenv("OBJ_DETECT_SQLITE_FILE", "obj_detect_dev.db")
            exec_db = f"sqlite:///{sqlite_file}"
        else:
            # Fallback to MySQL if mysql env values present
            user = getenv("OBJ_DETECT_MYSQL_USER", "")
            pwd = getenv("OBJ_DETECT_MYSQL_PWD", "")
            host = getenv("OBJ_DETECT_MYSQL_HOST", "localhost")
            db = getenv("OBJ_DETECT_MYSQL_DB", "")
            if not (user and pwd and db):
                raise RuntimeError(
                    "No DATABASE_URL and MySQL settings incomplete. "
                    "Set DATABASE_URL or OBJ_DETECT_USE_SQLITE=1 for development."
                )
            exec_db = f"mysql+mysqldb://{user}:{pwd}@{host}/{db}"

        # create engine
        # For sqlite we need connect_args to allow check_same_thread when using scoped sessions in some scenarios.
        connect_args = {}
        if exec_db.startswith("sqlite"):
            connect_args = {"check_same_thread": False}

        self._engine = create_engine(exec_db, pool_pre_ping=True, connect_args=connect_args)

        # if test environment create/drop tables automatically on initialization
        if env == "test":
            Base.metadata.drop_all(self._engine)
            Base.metadata.create_all(self._engine)

    # --------------------
    # Session management
    # --------------------
    def reload(self) -> None:
        """Create all tables (if missing) and instantiate a session."""
        Base.metadata.create_all(self._engine)
        SessionFactory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._scoped = scoped_session(SessionFactory)
        self._session = self._scoped()

    @property
    def session(self):
        """Return the active SQLAlchemy Session instance. Call reload() first."""
        if self._session is None:
            # lazy-create session if not initialized
            self.reload()
        return self._session

    def close(self) -> None:
        """Close the scoped session and dispose engine connection."""
        if getattr(self, "_scoped", None):
            self._scoped.remove()
        if self._session:
            self._session.close()
            self._session = None

    # --------------------
    # Convenience helpers
    # --------------------
    def new(self, obj) -> None:
        """Add object to current session (does not commit)."""
        self.session.add(obj)

    def save(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def delete(self, obj) -> None:
        """Delete object from session and commit."""
        self.session.delete(obj)
        self.save()

    def get(self, cls, id: Optional[str] = None, **filters):
        """Retrieve a single object by id or additional filters.

        Examples:
            get(AIModel, id="...")  # by id
            get(ModelLabel, label_id="...", ai_model_id="...") # by filters
        """
        q = self.session.query(cls)
        if id:
            return q.filter_by(id=id).one_or_none()
        if filters:
            return q.filter_by(**filters).one_or_none()
        return None

    def all(self, cls=None):
        """Return all rows for a class or None if cls not provided."""
        if cls:
            return self.session.query(cls).all()
        return None

    def update(self, cls, id: str, **kwargs):
        """Update an object by id with provided kwargs. Returns updated object or None."""
        obj = self.get(cls, id=id)
        if not obj:
            return None
        for k, v in kwargs.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        # update timestamp if available
        if hasattr(obj, "updated_at"):
            obj.updated_at = datetime.now()
        self.save()
        return obj
