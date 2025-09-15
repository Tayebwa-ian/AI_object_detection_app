#!/usr/bin/python3
"""ModelLabel - aggregated metrics for a (AIModel, Label) pair, optionally linked to a run."""

from sqlalchemy import Column, Float, String, ForeignKey, Index
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel


class ModelLabel(BaseModel, Base):
    """Metrics per ai_model & label, optionally associated to an EvaluationRun.

    Columns:
        accuracy, precision, recall, f1_score - floats in [0.0, 1.0]
        ai_model_id, label_id - foreign keys
        run_id - optional FK to EvaluationRun to group metrics by run
    """
    __tablename__ = "models_labels"

    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)

    ai_model_id = Column(String(60), ForeignKey("ai_models.id"), nullable=False, index=True)
    label_id = Column(String(60), ForeignKey("labels.id"), nullable=False, index=True)
    run_id = Column(String(60), ForeignKey("evaluation_runs.id"), nullable=True, index=True)

    # relationships
    ai_model = relationship("AIModel", back_populates="model_labels")
    label = relationship("Label", back_populates="model_labels")
    run = relationship("EvaluationRun", back_populates="model_labels")

    # helpful index to quickly find metrics by (ai_model_id, label_id, run_id)
    __table_args__ = (
        Index("ix_model_label_model_label_run", "ai_model_id", "label_id", "run_id"),
    )

    def __init__(self, ai_model_id: str = None, label_id: str = None, run_id: str = None, **kwargs):
        super().__init__(**kwargs)
        if ai_model_id:
            self.ai_model_id = ai_model_id
        if label_id:
            self.label_id = label_id
        if run_id:
            self.run_id = run_id
