#!/usr/bin/python3
"""Output - model predictions (one row per detected object / prediction).

This model stores individual predictions produced by an AI model for a given Input.
It links to Input, Label and AIModel via foreign keys and provides fields commonly
needed for metrics (confidence, counts, bbox), plus an optional human verification flag.

Design notes:
 - FK columns are indexed for faster joins/aggregations.
 - Relationships use `back_populates` so queries can navigate both directions efficiently.
 - Keep bbox as text (JSON string) so you can store flexible geometry data without schema migrations.
"""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel


class Output(BaseModel, Base):
    """Outputs table: one prediction detected for an Input by an AIModel."""

    __tablename__ = "outputs"

    # Foreign keys (indexed for faster joins)
    input_id = Column(String(60), ForeignKey("inputs.id"), nullable=False, index=True)
    label_id = Column(String(60), ForeignKey("labels.id"), nullable=False, index=True)
    ai_model_id = Column(String(60), ForeignKey("ai_models.id"), nullable=False, index=True)

    # Prediction fields
    predicted_count = Column(Integer, nullable=False, default=1)
    corrected_count = Column(Integer, default=0)  # user corrected count if any
    confidence = Column(Float, default=0.0)       # normalized confidence (0.0 - 1.0)
    bbox = Column(Text)                            # JSON string describing bounding box/geometry
    is_human_verified = Column(Boolean, default=False)

    # Relationships (back_populates must match the other-side relationship names)
    input = relationship("Input", back_populates="outputs")
    label = relationship("Label", back_populates="outputs")
    ai_model = relationship("AIModel", back_populates="outputs")

    def __init__(
        self,
        input_id: str = None,
        label_id: str = None,
        ai_model_id: str = None,
        predicted_count: int = 1,
        confidence: float = 0.0,
        bbox: str = None,
        corrected_count: int = 0,
        is_human_verified: bool = False,
        **kwargs
    ):
        """Initialize Output with optional convenience parameters."""
        super().__init__(**kwargs)
        if input_id:
            self.input_id = input_id
        if label_id:
            self.label_id = label_id
        if ai_model_id:
            self.ai_model_id = ai_model_id
        self.predicted_count = predicted_count
        self.corrected_count = corrected_count
        self.confidence = confidence
        if bbox is not None:
            self.bbox = bbox
        self.is_human_verified = is_human_verified
