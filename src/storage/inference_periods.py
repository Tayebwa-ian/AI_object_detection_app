#!/usr/bin/python3
"""InferencePeriod - timing records (latency) for inferences, optionally linked to runs."""

from sqlalchemy import Column, String, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel


class InferencePeriod(BaseModel, Base):
    """Records inference time for (ai_model, input), optionally associated to a run."""
    __tablename__ = "inference_periods"

    value = Column(Float, default=0.0)  # seconds (or ms) depending on convention
    ai_model_id = Column(String(60), ForeignKey("ai_models.id"), nullable=False, index=True)
    input_id = Column(String(60), ForeignKey("inputs.id"), nullable=False, index=True)
    run_id = Column(String(60), ForeignKey("evaluation_runs.id"), nullable=True, index=True)

    ai_model = relationship("AIModel", back_populates="inference_periods")
    input = relationship("Input", back_populates="inference_periods")
    run = relationship("EvaluationRun", back_populates="inference_periods")

    __table_args__ = (
        Index("ix_inference_period_ai_model_input_run", "ai_model_id", "input_id", "run_id"),
    )

    def __init__(self, value: float = 0.0, ai_model_id: str = None, input_id: str = None, run_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        if ai_model_id:
            self.ai_model_id = ai_model_id
        if input_id:
            self.input_id = input_id
        if run_id:
            self.run_id = run_id
