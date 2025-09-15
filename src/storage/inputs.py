#!/usr/bin/python3
"""Input - table for inference inputs (images, prompts, flags)"""
from sqlalchemy import Column, String, Text, Integer, Boolean
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel


class Input(BaseModel, Base):
    """Input rows represent what was fed into models (image + prompt).

    Columns:
        prompt (Text)
        image_path (str) unique identifier for the image on disk or storage
        violation_count (int)
        is_zero_shot (bool)
        is_few_shot (bool)
        is_test (bool)
    Relationships:
        outputs -> Output predictions generated for this input
        inference_periods -> InferencePeriod timing records for this input
    """
    __tablename__ = "inputs"

    prompt = Column(Text)
    image_path = Column(String(200), nullable=False, unique=True, index=True)
    violation_count = Column(Integer, default=0)
    is_zero_shot = Column(Boolean, default=False)
    is_few_shot = Column(Boolean, default=False)
    is_test = Column(Boolean, default=False)

    outputs = relationship("Output", back_populates="input", cascade="all, delete-orphan")
    inference_periods = relationship("InferencePeriod", back_populates="input", cascade="all, delete-orphan")

    def __init__(self, image_path: str = None, prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        if image_path:
            self.image_path = image_path
        if prompt:
            self.prompt = prompt
