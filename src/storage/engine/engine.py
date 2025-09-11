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
        if OBJ_DETECT_ENV != 'test':
            exec_db = f'mysql+mysqldb://{OBJ_DETECT_MYSQL_USER}:{OBJ_DETECT_MYSQL_PWD}@{OBJ_DETECT_MYSQL_HOST}/{OBJ_DETECT_MYSQL_DB}'
        else:  # Configure an SQLITE DB instance for testing
            exec_db = f'sqlite:///{OBJ_DETECT_MYSQL_DB}'
        # Create the engine
        self.__engine = create_engine(exec_db, pool_pre_ping=True)

        if OBJ_DETECT_ENV == 'test':
            # Drop all tables to ensure a clean slate for testing
            Base.metadata.drop_all(self.__engine)
            Base.metadata.create_all(self.__engine)  # Recreate tables for testing

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
        """retrieve one object based on cls and id
        Args:
            cls: class of the object
            id: Id of the object
        Return: object based on the class and its ID, or None
        """
        if id:
            query = self.__session.query(cls).\
                filter_by(id=id).one_or_none()
            return query
        
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
