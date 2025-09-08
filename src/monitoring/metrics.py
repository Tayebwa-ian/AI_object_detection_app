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
        # For simplicity, we'll use a basic accuracy calculation
        # In a real implementation, you'd want to compute these from historical data
        
        # Calculate accuracy (how close the prediction was)
        if corrected_count > 0:
            accuracy = 1.0 - abs(predicted_count - corrected_count) / max(corrected_count, 1)
        else:
            accuracy = 1.0 if predicted_count == 0 else 0.0
            
        # For precision/recall/F1, we'd need more sophisticated logic
        # For now, we'll use accuracy as a proxy
        precision = accuracy
        recall = accuracy
        f1_score = accuracy
        
        # Update metrics
        quality_metrics['accuracy'].labels(object_type=object_type).set(accuracy)
        quality_metrics['precision'].labels(object_type=object_type).set(precision)
        quality_metrics['recall'].labels(object_type=object_type).set(recall)
        quality_metrics['f1_score'].labels(object_type=object_type).set(f1_score)
        
        logger.debug(f"Updated quality metrics for {object_type}: accuracy={accuracy:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to update quality metrics: {e}")

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
