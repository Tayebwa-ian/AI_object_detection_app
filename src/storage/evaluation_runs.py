#!/usr/bin/python3
"""EvaluationRun model.

Represents a single evaluation or training run for an AI model. This lets you
store grouped metrics (ModelLabel, InferencePeriod) belonging to a single run.
"""
from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from .base_model import Base, BaseModel


class EvaluationRun(BaseModel, Base):
    """Record of a training/evaluation run for an AIModel.

    Columns:
        ai_model_id (str): FK to AIModel
        run_type (str): 'train' or 'test'
        started_at (datetime)
        finished_at (datetime)
        meta_data (Text): optional JSON or text describing dataset/params
    Relationships:
        ai_model: the AIModel this run belongs to
        model_labels: aggregated metrics produced during this run
        inference_periods: latency records produced during this run
    """
    __tablename__ = "evaluation_runs"

    ai_model_id = Column(String(60), ForeignKey("ai_models.id"), nullable=False, index=True)
    run_type = Column(String(32), nullable=False)  # 'train' | 'test'
    started_at = Column(DateTime, default=datetime.now(), nullable=False)
    finished_at = Column(DateTime)
    meta_data = Column(Text)  # store parameters or dataset info as JSON/text

    # relationships
    ai_model = relationship("AIModel", back_populates="evaluation_runs")
    model_labels = relationship("ModelLabel", back_populates="run", cascade="all, delete-orphan")
    inference_periods = relationship("InferencePeriod", back_populates="run", cascade="all, delete-orphan")

    def __init__(self, ai_model_id: str = None, run_type: str = "test", metadata: str = None, **kwargs):
        """Initialize an EvaluationRun."""
        super().__init__(**kwargs)
        if ai_model_id:
            self.ai_model_id = ai_model_id
        self.run_type = run_type
        if metadata is not None:
            self.metadata = metadata
        self.started_at = datetime.now()
