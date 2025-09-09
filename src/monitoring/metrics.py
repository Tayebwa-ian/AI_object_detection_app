"""
Prometheus Metrics Collection for AI Object Counting Application

This module defines and manages all Prometheus metrics for monitoring
application performance, model inference times, and quality metrics.
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)

logger = logging.getLogger(__name__)

# Create a custom registry for our metrics
METRICS_REGISTRY = CollectorRegistry()

# =============================================================================
# RESPONSE TIME METRICS
# =============================================================================

# Overall application response time
app_response_time = Histogram(
    'app_response_seconds',
    'Time spent processing API requests',
    ['endpoint', 'method'],
    registry=METRICS_REGISTRY,
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float('inf'))
)

# Model inference times
model_inference_time = Histogram(
    'model_inference_seconds',
    'Time spent on model inference',
    ['model', 'object_type'],
    registry=METRICS_REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
)

# Few-shot learning specific metrics
fewshot_training_time = Histogram(
    'fewshot_training_seconds',
    'Time spent on few-shot learning training',
    ['object_type'],
    registry=METRICS_REGISTRY,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf'))
)

fewshot_support_images_count = Gauge(
    'fewshot_support_images_total',
    'Number of support images used for few-shot learning',
    ['object_type'],
    registry=METRICS_REGISTRY
)

fewshot_prototype_dimension = Gauge(
    'fewshot_prototype_dimension',
    'Dimension of few-shot learning prototypes',
    ['object_type'],
    registry=METRICS_REGISTRY
)

# =============================================================================
# MODEL CONFIDENCE METRICS
# =============================================================================

# Model confidence per object type
model_confidence = Histogram(
    'model_confidence',
    'Model confidence scores',
    ['model', 'object_type'],
    registry=METRICS_REGISTRY,
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# =============================================================================
# QUALITY METRICS (computed from database corrections)
# =============================================================================

quality_metrics = {
    'accuracy': Gauge(
        'model_accuracy',
        'Model accuracy by object type',
        ['object_type'],
        registry=METRICS_REGISTRY
    ),
    'precision': Gauge(
        'model_precision', 
        'Model precision by object type',
        ['object_type'],
        registry=METRICS_REGISTRY
    ),
    'recall': Gauge(
        'model_recall',
        'Model recall by object type', 
        ['object_type'],
        registry=METRICS_REGISTRY
    ),
    'f1_score': Gauge(
        'model_f1_score',
        'Model F1 score by object type',
        ['object_type'],
        registry=METRICS_REGISTRY
    )
}

# =============================================================================
# METADATA METRICS (per-request information)
# =============================================================================

metadata_metrics = {
    'image_width': Gauge(
        'image_width_pixels',
        'Width of processed images in pixels',
        registry=METRICS_REGISTRY
    ),
    'image_height': Gauge(
        'image_height_pixels', 
        'Height of processed images in pixels',
        registry=METRICS_REGISTRY
    ),
    'predicted_count': Gauge(
        'predicted_object_count',
        'Number of objects predicted by the model',
        ['object_type'],
        registry=METRICS_REGISTRY
    ),
    'segments_total': Gauge(
        'segments_found_total',
        'Total number of segments found in images',
        registry=METRICS_REGISTRY
    ),
    'object_types_found': Gauge(
        'object_types_found_total',
        'Number of different object types found in images',
        registry=METRICS_REGISTRY
    ),
    'avg_segment_area': Gauge(
        'avg_segment_area_pixels',
        'Average area of segments in pixels',
        registry=METRICS_REGISTRY
    ),
    'pipeline_models_used': Info(
        'pipeline_models_info',
        'Information about models used in the pipeline',
        registry=METRICS_REGISTRY
    )
}

# =============================================================================
# REQUEST COUNTERS
# =============================================================================

request_counter = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status'],
    registry=METRICS_REGISTRY
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def update_quality_metrics(object_type: str, predicted_count: int, corrected_count: int) -> None:
    """
    Update quality metrics based on predicted vs corrected counts.
    
    Args:
        object_type: Type of object being counted
        predicted_count: Count predicted by the model
        corrected_count: Count corrected by the user
    """
    try:
        # Calculate accuracy (how close the prediction was)
        if corrected_count > 0:
            accuracy = 1.0 - abs(predicted_count - corrected_count) / max(corrected_count, 1)
        else:
            accuracy = 1.0 if predicted_count == 0 else 0.0
        
        # Calculate precision: True Positives / (True Positives + False Positives)
        # For counting tasks, this is how many of our predictions were correct
        if predicted_count > 0:
            precision = min(predicted_count, corrected_count) / predicted_count
        else:
            precision = 1.0 if corrected_count == 0 else 0.0
            
        # Calculate recall: True Positives / (True Positives + False Negatives)  
        # For counting tasks, this is how many of the actual objects we found
        if corrected_count > 0:
            recall = min(predicted_count, corrected_count) / corrected_count
        else:
            recall = 1.0 if predicted_count == 0 else 0.0
            
        # Calculate F1 score: harmonic mean of precision and recall
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Update metrics
        quality_metrics['accuracy'].labels(object_type=object_type).set(accuracy)
        quality_metrics['precision'].labels(object_type=object_type).set(precision)
        quality_metrics['recall'].labels(object_type=object_type).set(recall)
        quality_metrics['f1_score'].labels(object_type=object_type).set(f1_score)
        
        logger.debug(f"Updated quality metrics for {object_type}: accuracy={accuracy:.3f}, precision={precision:.3f}, recall={recall:.3f}, f1={f1_score:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to update quality metrics: {e}")

def compute_quality_metrics_from_database(object_type: str = None) -> Dict[str, float]:
    """
    Compute quality metrics from all database corrections.
    
    Args:
        object_type: Optional filter for specific object type
        
    Returns:
        Dictionary with computed metrics
    """
    try:
        from ..storage import database, Output
        
        # Get all outputs with corrections
        query = database.session.query(Output).filter(Output.corrected_count.isnot(None))
        if object_type:
            # Join with ObjectType to filter by object type name
            from ..storage import ObjectType
            query = query.join(ObjectType).filter(ObjectType.name == object_type)
        
        outputs = query.all()
        
        if not outputs:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Calculate metrics across all corrections
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        count = 0
        
        for output in outputs:
            predicted = output.predicted_count or 0
            corrected = output.corrected_count or 0
            
            # Calculate per-sample metrics
            if corrected > 0:
                accuracy = 1.0 - abs(predicted - corrected) / max(corrected, 1)
            else:
                accuracy = 1.0 if predicted == 0 else 0.0
                
            if predicted > 0:
                precision = min(predicted, corrected) / predicted
            else:
                precision = 1.0 if corrected == 0 else 0.0
                
            if corrected > 0:
                recall = min(predicted, corrected) / corrected
            else:
                recall = 1.0 if predicted == 0 else 0.0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1
        
        # Return averages
        return {
            'accuracy': total_accuracy / count if count > 0 else 0.0,
            'precision': total_precision / count if count > 0 else 0.0,
            'recall': total_recall / count if count > 0 else 0.0,
            'f1_score': total_f1 / count if count > 0 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Failed to compute quality metrics from database: {e}")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

def record_request_metadata(
    image_width: int,
    image_height: int, 
    predicted_count: int,
    object_type: str,
    segments_count: int,
    object_types_found: int,
    avg_segment_area: float,
    models_used: list
) -> None:
    """
    Record metadata about a processed request.
    
    Args:
        image_width: Width of the processed image
        image_height: Height of the processed image
        predicted_count: Number of objects predicted
        object_type: Type of object being counted
        segments_count: Number of segments found
        object_types_found: Number of different object types found
        avg_segment_area: Average area of segments
        models_used: List of models used in the pipeline
    """
    try:
        # Update metadata metrics
        metadata_metrics['image_width'].set(image_width)
        metadata_metrics['image_height'].set(image_height)
        metadata_metrics['predicted_count'].labels(object_type=object_type).set(predicted_count)
        metadata_metrics['segments_total'].set(segments_count)
        metadata_metrics['object_types_found'].set(object_types_found)
        metadata_metrics['avg_segment_area'].set(avg_segment_area)
        
        # Update pipeline info
        metadata_metrics['pipeline_models_used'].info({
            'models': ','.join(models_used),
            'pipeline_version': '1.0'
        })
        
        logger.debug(f"Recorded metadata: {image_width}x{image_height}, {predicted_count} {object_type}s, {segments_count} segments")
        
    except Exception as e:
        logger.error(f"Failed to record request metadata: {e}")

def record_model_inference(model_name: str, object_type: str, duration: float, confidence: float = None) -> None:
    """
    Record model inference timing and confidence.
    
    Args:
        model_name: Name of the model (sam, resnet, mapper)
        object_type: Type of object being processed
        duration: Inference time in seconds
        confidence: Model confidence score (optional)
    """
    try:
        # Record inference time
        model_inference_time.labels(model=model_name, object_type=object_type).observe(duration)
        
        # Record confidence if provided
        if confidence is not None:
            model_confidence.labels(model=model_name, object_type=object_type).observe(confidence)
            
        logger.debug(f"Recorded {model_name} inference: {duration:.3f}s, confidence={confidence}")
        
    except Exception as e:
        logger.error(f"Failed to record model inference: {e}")

def record_api_request(endpoint: str, method: str, status_code: int) -> None:
    """
    Record API request metrics.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        status_code: HTTP status code
    """
    try:
        request_counter.labels(endpoint=endpoint, method=method, status=status_code).inc()
        logger.debug(f"Recorded API request: {method} {endpoint} -> {status_code}")
        
    except Exception as e:
        logger.error(f"Failed to record API request: {e}")

def get_metrics() -> str:
    """
    Generate OpenMetrics format metrics.
    
    Returns:
        String containing metrics in OpenMetrics format
    """
    try:
        return generate_latest(METRICS_REGISTRY)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return ""

def get_metrics_content_type() -> str:
    """
    Get the content type for metrics endpoint.
    
    Returns:
        Content type string for OpenMetrics
    """
    return CONTENT_TYPE_LATEST

def refresh_quality_metrics_from_database() -> None:
    """
    Refresh quality metrics by computing them from all database corrections.
    This can be called periodically to update metrics based on historical data.
    """
    try:
        from ..storage import database, Output, ObjectType
        
        # Get all object types that have corrections
        query = database.session.query(ObjectType).join(Output).filter(
            Output.corrected_count.isnot(None)
        ).distinct()
        
        object_types = query.all()
        
        for obj_type in object_types:
            # Compute metrics for this object type
            metrics = compute_quality_metrics_from_database(obj_type.name)
            
            # Update the metrics
            quality_metrics['accuracy'].labels(object_type=obj_type.name).set(metrics['accuracy'])
            quality_metrics['precision'].labels(object_type=obj_type.name).set(metrics['precision'])
            quality_metrics['recall'].labels(object_type=obj_type.name).set(metrics['recall'])
            quality_metrics['f1_score'].labels(object_type=obj_type.name).set(metrics['f1_score'])
            
            logger.debug(f"Refreshed quality metrics for {obj_type.name}: {metrics}")
            
    except Exception as e:
        logger.error(f"Failed to refresh quality metrics from database: {e}")

def record_fewshot_training(object_type: str, training_time: float, support_images_count: int, prototype_dimension: int) -> None:
    """
    Record few-shot learning training metrics.
    
    Args:
        object_type: Name of the object type being trained
        training_time: Training time in seconds
        support_images_count: Number of support images used
        prototype_dimension: Dimension of the learned prototype
    """
    try:
        fewshot_training_time.labels(object_type=object_type).observe(training_time)
        fewshot_support_images_count.labels(object_type=object_type).set(support_images_count)
        fewshot_prototype_dimension.labels(object_type=object_type).set(prototype_dimension)
        
        logger.debug(f"Recorded few-shot training for {object_type}: {training_time:.3f}s, {support_images_count} images, dim={prototype_dimension}")
        
    except Exception as e:
        logger.error(f"Failed to record few-shot training metrics: {e}")

def refresh_fewshot_metrics_from_database() -> None:
    """
    Refresh few-shot learning metrics from database.
    """
    try:
        from ..storage import database, FewShotObjectType
        
        # Get all few-shot object types
        fewshot_types = database.all(FewShotObjectType)
        
        for obj_type in fewshot_types:
            # Update metrics
            fewshot_support_images_count.labels(object_type=obj_type.name).set(obj_type.support_images_count)
            fewshot_prototype_dimension.labels(object_type=obj_type.name).set(obj_type.feature_dimension)
            
            logger.debug(f"Refreshed few-shot metrics for {obj_type.name}")
            
    except Exception as e:
        logger.error(f"Failed to refresh few-shot metrics from database: {e}")
