"""
Monitoring Decorators for Flask Application

This module provides decorators for automatically tracking API request
timing and other metrics.
"""

import time
import functools
import logging
from typing import Callable, Any
from flask import request, g

from .metrics import app_response_time, record_api_request

logger = logging.getLogger(__name__)

def track_response_time(endpoint_name: str = None):
    """
    Decorator to track API response time.
    
    Args:
        endpoint_name: Custom name for the endpoint (defaults to request.endpoint)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Start timing
            start_time = time.time()
            
            # Get endpoint name
            endpoint = endpoint_name or request.endpoint or 'unknown'
            method = request.method
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Record metrics
                app_response_time.labels(endpoint=endpoint, method=method).observe(duration)
                record_api_request(endpoint, method, 200)
                
                # Store timing info in Flask g for potential use in response
                g.request_duration = duration
                
                logger.debug(f"Request {method} {endpoint} completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                # Calculate duration even for errors
                duration = time.time() - start_time
                
                # Record error metrics
                app_response_time.labels(endpoint=endpoint, method=method).observe(duration)
                record_api_request(endpoint, method, 500)
                
                logger.error(f"Request {method} {endpoint} failed after {duration:.3f}s: {e}")
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator

def track_model_inference(model_name: str):
    """
    Decorator to track model inference time.
    
    Args:
        model_name: Name of the model being used
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract object_type from kwargs or result if available
                object_type = kwargs.get('object_type', 'unknown')
                if hasattr(result, 'get') and isinstance(result, dict):
                    object_type = result.get('object_type', object_type)
                
                # Record model inference time
                from .metrics import record_model_inference
                record_model_inference(model_name, object_type, duration)
                
                logger.debug(f"Model {model_name} inference completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Model {model_name} inference failed after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator
