"""
Few-Shot Learning Service

This module provides the core few-shot learning functionality that integrates
with the existing AI Object Counter pipeline. It implements prototype-based
few-shot learning using ResNet features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import cv2
import os
from pathlib import Path

from transformers import AutoImageProcessor, AutoModelForImageClassification
from ..storage.fewshot_models import FewShotObjectType, FewShotSupportImage, FewShotPrediction, FewShotLearningSession
from ..storage import database
from ..monitoring.metrics import record_model_inference, record_request_metadata, record_fewshot_training

logger = logging.getLogger(__name__)


class ResNetFeatureExtractor:
    """Feature extractor using ResNet-50 backbone"""
    
    def __init__(self, model_name="microsoft/resnet-50", device="cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Remove the final classification layer to get features
        # Handle different model architectures
        if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'in_features'):
            self.feature_dim = self.model.classifier.in_features
            self.model.classifier = nn.Identity()  # Remove classification head
        elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'):
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()  # Remove classification head
        else:
            # Fallback: ResNet-50 feature dimension is 2048
            self.feature_dim = 2048
            if hasattr(self.model, 'classifier'):
                self.model.classifier = nn.Identity()
            elif hasattr(self.model, 'fc'):
                self.model.fc = nn.Identity()
        
        logger.info(f"ResNet-50 feature extractor loaded on {device}")
        logger.info(f"Feature dimension: {self.feature_dim}")
    
    def extract_features(self, image) -> np.ndarray:
        """Extract features from a single image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                features = outputs.logits
            else:
                features = outputs
            # Normalize features
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy().flatten()
    
    def extract_batch_features(self, images: List) -> np.ndarray:
        """Extract features from a batch of images"""
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)
        return np.array(features_list)


class PrototypeFewShotLearner:
    """Prototype-based few-shot learning system"""
    
    def __init__(self, feature_extractor: ResNetFeatureExtractor, distance_metric="cosine"):
        self.feature_extractor = feature_extractor
        self.distance_metric = distance_metric
        self.prototypes = {}
        self.class_names = []
        
    def compute_prototype(self, support_features: np.ndarray) -> np.ndarray:
        """Compute class prototype from support examples"""
        if self.distance_metric == "cosine":
            # For cosine similarity, use mean of normalized features
            prototype = np.mean(support_features, axis=0)
            prototype = prototype / np.linalg.norm(prototype)  # Normalize
        else:
            # For Euclidean distance, use simple mean
            prototype = np.mean(support_features, axis=0)
        return prototype
    
    def predict_with_confidence(self, query_image, fewshot_object_type: FewShotObjectType) -> Tuple[int, float, float]:
        """
        Predict class and return confidence for a single query image
        
        Args:
            query_image: Image to classify
            fewshot_object_type: Few-shot object type to classify against
            
        Returns:
            Tuple of (predicted_count, confidence_score, distance_to_prototype)
        """
        # Extract features from query image
        query_features = self.feature_extractor.extract_features(query_image)
        
        # Get prototype features
        prototype_features = fewshot_object_type.get_prototype_features()
        
        if len(prototype_features) == 0:
            raise ValueError(f"No prototype features found for {fewshot_object_type.name}")
        
        # Compute distance to prototype
        if self.distance_metric == "cosine":
            # Cosine similarity (higher is better)
            similarity = np.dot(query_features, prototype_features)
            distance = 1 - similarity  # Convert to distance
            confidence = similarity  # Use similarity as confidence
        else:
            # Euclidean distance
            distance = np.linalg.norm(query_features - prototype_features)
            confidence = 1.0 / (1.0 + distance)  # Convert distance to confidence
        
        # For now, we'll predict 1 if confidence is above threshold, 0 otherwise
        # In a more sophisticated system, you might use the confidence to predict actual counts
        predicted_count = 1 if confidence > 0.5 else 0
        
        return predicted_count, confidence, distance


class FewShotLearningService:
    """Main service for few-shot learning operations"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.feature_extractor = ResNetFeatureExtractor(device=device)
        self.learner = PrototypeFewShotLearner(self.feature_extractor, distance_metric="cosine")
        
    def register_object_type(self, 
                           object_name: str, 
                           support_images: List[np.ndarray], 
                           description: str = None) -> Dict[str, Any]:
        """
        Register a new object type using few-shot learning
        
        Args:
            object_name: Name of the new object type
            support_images: List of support images (numpy arrays)
            description: Optional description of the object type
            
        Returns:
            Dictionary with registration results
        """
        start_time = time.time()
        
        try:
            # Check if object type already exists
            existing = database.get(FewShotObjectType, name=object_name)
            if existing:
                return {
                    'success': False,
                    'error': f'Object type "{object_name}" already exists',
                    'object_type_id': existing.id
                }
            
            # Extract features from support images
            logger.info(f"Extracting features from {len(support_images)} support images...")
            feature_extraction_start = time.time()
            
            support_features = self.feature_extractor.extract_batch_features(support_images)
            
            feature_extraction_time = (time.time() - feature_extraction_start) * 1000
            
            # Compute prototype
            prototype = self.learner.compute_prototype(support_features)
            
            # Create few-shot object type
            fewshot_obj_type = FewShotObjectType(
                name=object_name,
                description=description or f"Few-shot learned object type: {object_name}",
                support_images_count=len(support_images)
            )
            fewshot_obj_type.set_prototype_features(prototype)
            
            # Save to database
            database.new(fewshot_obj_type)
            database.save()
            
            # Save support images
            for i, (image, features) in enumerate(zip(support_images, support_features)):
                # Save image to media directory
                image_filename = f"fewshot_{fewshot_obj_type.id}_{i}.jpg"
                image_path = self._save_support_image(image, image_filename)
                
                # Create support image record
                support_image = FewShotSupportImage(
                    fewshot_object_type_id=fewshot_obj_type.id,
                    image_path=image_path,
                    image_filename=image_filename,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                    image_size_bytes=image.nbytes
                )
                support_image.set_feature_vector(features)
                
                database.new(support_image)
            
            # Create learning session record
            training_time = (time.time() - start_time) * 1000
            learning_session = FewShotLearningSession(
                fewshot_object_type_id=fewshot_obj_type.id,
                session_type='initial_training',
                support_images_count=len(support_images),
                feature_extractor_model='microsoft/resnet-50',
                distance_metric='cosine',
                training_time_ms=training_time,
                session_metadata={
                    'feature_extraction_time_ms': feature_extraction_time,
                    'prototype_dimension': len(prototype)
                }
            )
            database.new(learning_session)
            database.save()
            
            # Record metrics
            record_model_inference('fewshot_training', object_name, training_time / 1000.0)
            record_fewshot_training(object_name, training_time / 1000.0, len(support_images), len(prototype))
            
            logger.info(f"Successfully registered few-shot object type: {object_name}")
            
            return {
                'success': True,
                'object_type_id': fewshot_obj_type.id,
                'object_name': object_name,
                'support_images_count': len(support_images),
                'training_time_ms': training_time,
                'feature_extraction_time_ms': feature_extraction_time,
                'prototype_dimension': len(prototype)
            }
            
        except Exception as e:
            logger.error(f"Failed to register few-shot object type: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def count_objects_fewshot(self, 
                            image: np.ndarray, 
                            object_name: str,
                            input_id: str = None) -> Dict[str, Any]:
        """
        Count objects using few-shot learning
        
        Args:
            image: Image to process (numpy array)
            object_name: Name of the few-shot object type
            input_id: Optional input ID for linking
            
        Returns:
            Dictionary with counting results
        """
        start_time = time.time()
        
        try:
            # Get few-shot object type
            logger.debug(f"Looking up few-shot object type: {object_name}")
            fewshot_obj_type = database.get(FewShotObjectType, name=object_name)
            if not fewshot_obj_type:
                return {
                    'success': False,
                    'error': f'Few-shot object type "{object_name}" not found'
                }
            
            if not fewshot_obj_type.is_active:
                return {
                    'success': False,
                    'error': f'Few-shot object type "{object_name}" is inactive'
                }
            
            logger.debug(f"Found active few-shot object type: {object_name}")
            
            # Extract features and predict
            logger.debug("Starting feature extraction and prediction")
            feature_extraction_start = time.time()
            predicted_count, confidence, distance = self.learner.predict_with_confidence(
                image, fewshot_obj_type
            )
            feature_extraction_time = (time.time() - feature_extraction_start) * 1000
            
            classification_time = (time.time() - feature_extraction_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Prediction completed: count={predicted_count}, confidence={confidence}")
            
            # Save prediction record
            logger.debug("Saving query image")
            image_filename = f"fewshot_query_{int(time.time())}.jpg"
            image_path = self._save_query_image(image, image_filename)
            
            logger.debug("Creating prediction record")
            prediction = FewShotPrediction(
                fewshot_object_type_id=fewshot_obj_type.id,
                input_id=input_id,
                predicted_count=int(predicted_count),
                confidence_score=float(confidence),
                distance_to_prototype=float(distance),
                image_path=image_path,
                image_width=int(image.shape[1]),
                image_height=int(image.shape[0]),
                processing_time_ms=float(total_time),
                feature_extraction_time_ms=float(feature_extraction_time),
                classification_time_ms=float(classification_time)
            )
            logger.debug("Saving prediction to database")
            database.new(prediction)
            database.save()
            logger.debug("Prediction saved successfully")
            
            # Record metrics
            record_model_inference('fewshot_classification', object_name, total_time / 1000.0, confidence)
            record_request_metadata(
                image_width=image.shape[1],
                image_height=image.shape[0],
                predicted_count=predicted_count,
                object_type=object_name,
                segments_count=1,  # Few-shot doesn't use segmentation
                object_types_found=1,
                avg_segment_area=image.shape[0] * image.shape[1],
                models_used=['fewshot_resnet']
            )
            
            return {
                'success': True,
                'predicted_count': int(predicted_count),
                'confidence': float(confidence),
                'distance_to_prototype': float(distance),
                'object_type': object_name,
                'processing_time_ms': float(total_time),
                'feature_extraction_time_ms': float(feature_extraction_time),
                'classification_time_ms': float(classification_time),
                'prediction_id': prediction.id
            }
            
        except Exception as e:
            logger.error(f"Failed to count objects using few-shot learning: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_fewshot_object_types(self) -> List[Dict[str, Any]]:
        """Get all registered few-shot object types"""
        try:
            object_types = database.all(FewShotObjectType)
            return [obj_type.to_dict() for obj_type in object_types]
        except Exception as e:
            logger.error(f"Failed to get few-shot object types: {e}")
            return []
    
    def get_fewshot_object_type(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific few-shot object type"""
        try:
            obj_type = database.get(FewShotObjectType, name=object_name)
            return obj_type.to_dict() if obj_type else None
        except Exception as e:
            logger.error(f"Failed to get few-shot object type {object_name}: {e}")
            return None
    
    def delete_fewshot_object_type(self, object_name: str) -> Dict[str, Any]:
        """Delete a few-shot object type"""
        try:
            obj_type = database.get(FewShotObjectType, name=object_name)
            if not obj_type:
                return {
                    'success': False,
                    'error': f'Few-shot object type "{object_name}" not found'
                }
            
            # Delete associated files
            for support_image in obj_type.support_images:
                self._delete_image_file(support_image.image_path)
            
            # Delete from database (cascade will handle related records)
            database.delete(obj_type)
            
            logger.info(f"Deleted few-shot object type: {object_name}")
            
            return {
                'success': True,
                'message': f'Few-shot object type "{object_name}" deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to delete few-shot object type: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_support_image(self, image: np.ndarray, filename: str) -> str:
        """Save support image to media directory"""
        from ..config import config
        
        # Ensure media directory exists
        media_dir = Path(config.MEDIA_DIRECTORY)
        fewshot_dir = media_dir / "fewshot"
        fewshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_path = fewshot_dir / filename
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return str(image_path.relative_to(media_dir))
    
    def _save_query_image(self, image: np.ndarray, filename: str) -> str:
        """Save query image to media directory"""
        try:
            from ..config import config
            
            # Ensure media directory exists
            media_dir = Path(config.MEDIA_DIRECTORY)
            fewshot_dir = media_dir / "fewshot"
            fewshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            image_path = fewshot_dir / filename
            success = cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            if not success:
                raise Exception(f"Failed to save image to {image_path}")
            
            return str(image_path.relative_to(media_dir))
        except Exception as e:
            logger.error(f"Failed to save query image: {e}")
            raise
    
    def _delete_image_file(self, image_path: str):
        """Delete image file from filesystem"""
        try:
            from ..config import config
            full_path = Path(config.MEDIA_DIRECTORY) / image_path
            if full_path.exists():
                full_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete image file {image_path}: {e}")


# Global service instance (lazy-fail safe)
try:
    fewshot_service = FewShotLearningService(device="cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    logger.error(f"FewShot service initialization failed: {e}")

    class _FewShotServiceUnavailable:
        """Fallback service that reports unavailability but allows app startup."""

        def __getattr__(self, name):
            def _fn(*args, **kwargs):
                return {
                    'success': False,
                    'error': 'Few-shot service unavailable: initialization failed'
                }
            return _fn

    fewshot_service = _FewShotServiceUnavailable()
