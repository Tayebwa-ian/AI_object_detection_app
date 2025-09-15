#!/usr/bin/python3
"""Label - table for object/label categories"""
from sqlalchemy import Column, String, Text
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel


class Label(BaseModel, Base):
    """Labels define the categories (e.g. 'dog','car','person').

    Columns:
        name (str): unique label name
        description (str)
    Relationships:
        outputs -> Output rows that detected this label
        model_labels -> aggregated metrics (ModelLabel) for each AI model vs this label
    """
    __tablename__ = "labels"

    name = Column(String(128), nullable=False, unique=True, index=True)
    description = Column(Text)

    outputs = relationship("Output", back_populates="label", cascade="all, delete-orphan")
    model_labels = relationship("ModelLabel", back_populates="label", cascade="all, delete-orphan")

    def __init__(self, name: str = None, description: str = None, **kwargs):
        super().__init__(**kwargs)
        if name:
            self.name = name
        if description:
            self.description = description
