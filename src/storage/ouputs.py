#!/usr/bin/python3"
"""Input Model - Module"""
from sqlalchemy import String, Column, Integer, Float, ForeignKey
from .base_model import Base, BaseModel

class Output(BaseModel, Base):
    """Creating an Outputs table in the database
    Args
        predicted_count: count of specific objects predicted from an image
        predicted_count: user correction of the actual count of the specific objects in an image
        pred_confidence: the percentage of which the model is confident of the the prediction
        object_type_id: Foreign key to associate outputs with objects
        input_id: Foreign key to associate outputs with inputs
    """
    __tablename__ = 'outputs'
    predicted_count = Column(Integer, nullable=False)
    corrected_count = Column(Integer)
    pred_confidence = Column(Float(), nullable=False)
    object_type_id = Column(String(60), ForeignKey("object_types.id"), nullable=False)
    input_id = Column(String(60), ForeignKey("inputs.id"), nullable=False)

    def __init__(self):
        """initializes Output class"""
        super().__init__()
