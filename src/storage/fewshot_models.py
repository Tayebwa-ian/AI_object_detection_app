#!/usr/bin/python3
"""Few-Shot Learning Models - Module"""

from sqlalchemy import String, Column, Text, Integer, Float, LargeBinary, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .base_model import Base, BaseModel
import json
import numpy as np
from typing import Dict, List, Any, Optional


class FewShotObjectType(BaseModel, Base):
    """Few-shot learning object type table
    
    Stores information about custom object types learned through few-shot learning.
    These are object types that are not in the predefined ResNet classes.
    """
    __tablename__ = 'fewshot_object_types'
    
    name = Column(String(128), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    feature_dimension = Column(Integer, nullable=False, default=2048)  # ResNet-50 feature dim
    prototype_features = Column(LargeBinary, nullable=False)  # Serialized prototype vector
    support_images_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Integer, nullable=False, default=1)  # 1=active, 0=inactive
    
    # Relationships
    support_images = relationship("FewShotSupportImage", backref="fewshot_object_type", cascade="all, delete-orphan")
    predictions = relationship("FewShotPrediction", backref="fewshot_object_type", cascade="all, delete-orphan")
    
    def __init__(self, **kwargs):
        """Initialize FewShotObjectType"""
        super().__init__()
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def set_prototype_features(self, features: np.ndarray):
        """Set prototype features from numpy array"""
        self.prototype_features = features.astype(np.float32).tobytes()
        self.feature_dimension = len(features)
    
    def get_prototype_features(self) -> np.ndarray:
        """Get prototype features as numpy array"""
        if self.prototype_features:
            return np.frombuffer(self.prototype_features, dtype=np.float32)
        return np.array([])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'feature_dimension': self.feature_dimension,
            'support_images_count': self.support_images_count,
            'is_active': bool(self.is_active),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class FewShotSupportImage(BaseModel, Base):
    """Support images for few-shot learning
    
    Stores the support images used to learn a few-shot object type.
    """
    __tablename__ = 'fewshot_support_images'
    
    fewshot_object_type_id = Column(String(60), ForeignKey('fewshot_object_types.id'), nullable=False)
    image_path = Column(String(512), nullable=False)
    image_filename = Column(String(256), nullable=False)
    feature_vector = Column(LargeBinary, nullable=False)  # Serialized feature vector
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    image_size_bytes = Column(Integer, nullable=False)
    
    def __init__(self, **kwargs):
        """Initialize FewShotSupportImage"""
        super().__init__()
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def set_feature_vector(self, features: np.ndarray):
        """Set feature vector from numpy array"""
        self.feature_vector = features.astype(np.float32).tobytes()
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector as numpy array"""
        if self.feature_vector:
            return np.frombuffer(self.feature_vector, dtype=np.float32)
        return np.array([])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'fewshot_object_type_id': self.fewshot_object_type_id,
            'image_path': self.image_path,
            'image_filename': self.image_filename,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_size_bytes': self.image_size_bytes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class FewShotPrediction(BaseModel, Base):
    """Few-shot learning predictions
    
    Stores predictions made using few-shot learning for monitoring and evaluation.
    """
    __tablename__ = 'fewshot_predictions'
    
    fewshot_object_type_id = Column(String(60), ForeignKey('fewshot_object_types.id'), nullable=False)
    input_id = Column(String(60), ForeignKey('inputs.id'), nullable=True)  # Link to original input
    output_id = Column(String(60), ForeignKey('outputs.id'), nullable=True)  # Link to output if exists
    
    # Prediction details
    predicted_count = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=False)
    distance_to_prototype = Column(Float, nullable=False)
    
    # Image metadata
    image_path = Column(String(512), nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    
    # Processing metadata
    processing_time_ms = Column(Float, nullable=False)
    feature_extraction_time_ms = Column(Float, nullable=False)
    classification_time_ms = Column(Float, nullable=False)
    
    # Optional correction
    corrected_count = Column(Integer, nullable=True)
    is_corrected = Column(Integer, nullable=False, default=0)  # 1=corrected, 0=not corrected
    
    def __init__(self, **kwargs):
        """Initialize FewShotPrediction"""
        super().__init__()
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'fewshot_object_type_id': self.fewshot_object_type_id,
            'input_id': self.input_id,
            'output_id': self.output_id,
            'predicted_count': self.predicted_count,
            'confidence_score': self.confidence_score,
            'distance_to_prototype': self.distance_to_prototype,
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'processing_time_ms': self.processing_time_ms,
            'feature_extraction_time_ms': self.feature_extraction_time_ms,
            'classification_time_ms': self.classification_time_ms,
            'corrected_count': self.corrected_count,
            'is_corrected': bool(self.is_corrected),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class FewShotLearningSession(BaseModel, Base):
    """Few-shot learning training sessions
    
    Tracks when few-shot learning sessions were performed for monitoring.
    """
    __tablename__ = 'fewshot_learning_sessions'
    
    fewshot_object_type_id = Column(String(60), ForeignKey('fewshot_object_types.id'), nullable=False)
    session_type = Column(String(50), nullable=False)  # 'initial_training', 'retraining', 'validation'
    
    # Training parameters
    support_images_count = Column(Integer, nullable=False)
    feature_extractor_model = Column(String(100), nullable=False, default='microsoft/resnet-50')
    distance_metric = Column(String(20), nullable=False, default='cosine')
    
    # Performance metrics
    training_time_ms = Column(Float, nullable=False)
    validation_accuracy = Column(Float, nullable=True)
    validation_samples_count = Column(Integer, nullable=True)
    
    # Metadata
    session_metadata = Column(JSON, nullable=True)  # Additional session information
    
    def __init__(self, **kwargs):
        """Initialize FewShotLearningSession"""
        super().__init__()
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'fewshot_object_type_id': self.fewshot_object_type_id,
            'session_type': self.session_type,
            'support_images_count': self.support_images_count,
            'feature_extractor_model': self.feature_extractor_model,
            'distance_metric': self.distance_metric,
            'training_time_ms': self.training_time_ms,
            'validation_accuracy': self.validation_accuracy,
            'validation_samples_count': self.validation_samples_count,
            'session_metadata': self.session_metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


