#!/usr/bin/python3"
"""Object_type Model - Module"""
from sqlalchemy import String, Column, Float
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel

class ObjectType(BaseModel, Base):
    """Creating an Object_type table in the database
    Args
        name: name of the object type
        description: A simple description of the object
        accuracy: accuracy at which the label was predicted
        precision: precision at which the label was predicted
        recall: recall at which the label was predicted
    """
    __tablename__ = 'object_types'
    name = Column(String(128), nullable=False, unique=True)
    description = Column(String(128), nullable=False)
    accuracy = Column(Float(), nullable=False)
    precision = Column(Float(), nullable=False)
    recall = Column(Float(), nullable=False)
    outputs = relationship("Output", backref="object_output", cascade="all, delete-orphan")

    def __init__(self):
        """initializes Object_type class"""
        super().__init__()
