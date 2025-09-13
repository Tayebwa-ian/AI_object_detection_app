#!/usr/bin/python3"
"""Input Model - Module"""
from sqlalchemy import String, Column, Text, Float
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel

class Input(BaseModel, Base):
    """Creating an Input table in the database
    Args
        description: Description prompt to give to the model
        image_path: Path to the submitted image
        resnet_inference_time: inference time for ResNet model
        bart_inference_time: inference time for BART model
    """
    __tablename__ = 'inputs'
    description = Column(Text, nullable=False)
    image_path = Column(String(200), nullable=False, unique=True)
    sam_inference_time = Column(Float(), nullable=False)
    resnet_inference_time = Column(Float(), nullable=False)
    outputs = relationship("Output", backref="input_output", cascade="all, delete-orphan")

    def __init__(self):
        """initializes Input class"""
        super().__init__()
