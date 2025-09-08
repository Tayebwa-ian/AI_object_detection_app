#!/usr/bin/python3
"""Engine - Module"""
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from src.storage.base_model import Base
from os import getenv
from datetime import datetime


class Engine:
    """Set up a connection to a database"""

    __session = None
    __engine = None

    def __init__(self):
        """intialize the Engine"""
        # setup connection to MySQL
        OBJ_DETECT_MYSQL_USER = getenv('OBJ_DETECT_MYSQL_USER')
        OBJ_DETECT_MYSQL_PWD = getenv('OBJ_DETECT_MYSQL_PWD')
        OBJ_DETECT_MYSQL_HOST = getenv('OBJ_DETECT_MYSQL_HOST')
        OBJ_DETECT_MYSQL_DB = getenv('OBJ_DETECT_MYSQL_DB')
        OBJ_DETECT_ENV = getenv('OBJ_DETECT_ENV')
        
        if OBJ_DETECT_ENV == 'production':
            exec_db = f'mysql+mysqldb://{OBJ_DETECT_MYSQL_USER}:{OBJ_DETECT_MYSQL_PWD}@{OBJ_DETECT_MYSQL_HOST}/{OBJ_DETECT_MYSQL_DB}'
        else:  # Use SQLite for development and testing
            exec_db = f'sqlite:///{OBJ_DETECT_MYSQL_DB}'
        
        # Create the engine (special handling for SQLite in threaded dev server)
        if exec_db.startswith('sqlite:///'):
            self.__engine = create_engine(
                exec_db,
                pool_pre_ping=True,
                connect_args={"check_same_thread": False}
            )
        else:
            self.__engine = create_engine(exec_db, pool_pre_ping=True)
        
        # Initialize session
        session_db = sessionmaker(bind=self.__engine, expire_on_commit=False)
        Session = scoped_session(session_db)
        self.__session = Session()

        if OBJ_DETECT_ENV == 'test':
            # In test env, reset DB to a clean state each run
            Base.metadata.drop_all(self.__engine)
            Base.metadata.create_all(self.__engine)
        else:
            # In development/production, ensure tables exist but do not drop data
            Base.metadata.create_all(self.__engine)

    def new(self, obj):
        """
            Creating new instance in db storage
        """
        self.__session.add(obj)

    def save(self):
        """
            save to the db storage
        """
        self.__session.commit()

    def get(self, cls, id=None, **kwargs) -> object:
        """retrieve one object based on cls and id or kwargs
        Args:
            cls: class of the object
            id: Id of the object
            **kwargs: additional filter criteria
        Return: object based on the class and its ID or kwargs, or None
        """
        if id:
            query = self.__session.query(cls).\
                filter_by(id=id).one_or_none()
            return query
        elif kwargs:
            query = self.__session.query(cls).\
                filter_by(**kwargs).one_or_none()
            return query
        return None
        
    def all(self, cls=None):
        """ query on the current database session (self.__session)
        all objects depending of the class name"""
        if cls:
            q = self.__session.query(cls).all()
            return (q)

    def delete(self, obj=None):
        """
            Delete obj from db storage
        """
        if obj:
            self.__session.delete(obj)
        self.save()

    def reload(self):
        """
            create table in database
        """
        Base.metadata.create_all(self.__engine)
        session_db = sessionmaker(bind=self.__engine, expire_on_commit=False)
        Session = scoped_session(session_db)
        self.__session = Session()

    def close(self) -> None:
        """
            Closing the session
        """
        if self.__session:
            self.__session.close()

    def update(self, cls, id, **kwargs):
        """Update an object in the database
        Args:
            kwargs: a dictionary of fields to update and their new values
        """
        obj = self.get(cls, id)
        if kwargs:
            for field in kwargs.keys():
                if hasattr(obj, field):
                        setattr(obj, field, kwargs[field])
            obj.updated_at = datetime.now()
            self.save()
        return obj
