#!/usr/bin/python3"
"""Input Model - Module"""
from sqlalchemy import String, Column, Float, ForeignKey
from .base_model import Base, BaseModel

class Metric(BaseModel, Base):
    """Creating an Metric table in the database
    Args
        accuracy: accuracy of the prediction
        precision: precision of the prediction
        recall: recall of the prediction
        sam_inference_time: inference time for SAM model
        resnet_inference_time: inference time for ResNet model
        bart_inference_time: inference time for BART model
    """
    __tablename__ = 'metrics'
    accuracy = Column(Float(), nullable=False)   # FIXED spelling
    precision = Column(Float(), nullable=False)
    recall = Column(Float(), nullable=False)
    sam_inference_time = Column(Float(), nullable=False)
    resnet_inference_time = Column(Float(), nullable=False)
    bart_inference_time = Column(Float(), nullable=True)
    output_id = Column(String(60), ForeignKey("outputs.id"), nullable=False)
    object_type_id = Column(String(60), ForeignKey("object_types.id"), nullable=False)

    def __init__(self):
        """initializes Metric class"""
        super().__init__()
