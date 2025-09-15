#!/usr/bin/python3
"""AIModel - table for ML model metadata"""
from sqlalchemy import Column, String, Text
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel


class AIModel(BaseModel, Base):
    """AIModel table stores metadata about models being benchmarked.

    Relationships:
      - outputs: Output rows produced by this model
      - model_labels: aggregated metrics
      - inference_periods: latency records
      - evaluation_runs: training/test runs for this model
    """
    __tablename__ = "ai_models"

    name = Column(String(128), nullable=False, unique=True, index=True)
    description = Column(Text)

    outputs = relationship("Output", back_populates="ai_model", cascade="all, delete-orphan")
    model_labels = relationship("ModelLabel", back_populates="ai_model", cascade="all, delete-orphan")
    inference_periods = relationship("InferencePeriod", back_populates="ai_model", cascade="all, delete-orphan")
    evaluation_runs = relationship("EvaluationRun", back_populates="ai_model", cascade="all, delete-orphan")

    def __init__(self, name: str = None, description: str = None, **kwargs):
        super().__init__(**kwargs)
        if name:
            self.name = name
        if description:
            self.description = description
