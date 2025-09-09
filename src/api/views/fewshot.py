"""
Few-Shot Learning API Views

This module provides REST API endpoints for few-shot learning functionality,
including registering new object types and counting objects using few-shot classification.
"""

import os
import time
import logging
import cv2
import numpy as np
from flask import request, jsonify, make_response
from flask_restful import Resource
from werkzeug.utils import secure_filename
from typing import Dict, List, Any

from ...storage.fewshot_models import FewShotObjectType, FewShotSupportImage, FewShotPrediction
from ...storage import database
from ...pipeline.fewshot_service import fewshot_service
from ...api.utils.error_handlers import validate_file_upload, ValidationAPIError, create_error_response
from ...monitoring.decorators import track_response_time
from ...monitoring.metrics import record_api_request

logger = logging.getLogger(__name__)


class FewShotRegister(Resource):
    """
    Register new object type using few-shot learning
    
    POST /api/fewshot/register
    - Register a new object type with support images
    - Learn prototype features from support examples
    """
    
    @track_response_time('api_fewshot_register')
    def post(self):
        """
        Register new object type with few-shot learning
        ---
        tags:
          - Few-Shot Learning
        parameters:
          - in: formData
            name: object_name
            type: string
            required: true
            description: Name of the new object type
          - in: formData
            name: description
            type: string
            required: false
            description: Description of the object type
          - in: formData
            name: support_images
            type: file
            required: true
            description: Support images for few-shot learning (multiple files)
        responses:
          201:
            description: Object type registered successfully
            schema:
              type: object
              properties:
                success:
                  type: boolean
                object_type_id:
                  type: string
                object_name:
                  type: string
                support_images_count:
                  type: integer
                training_time_ms:
                  type: number
                feature_extraction_time_ms:
                  type: number
                prototype_dimension:
                  type: integer
          400:
            description: Bad request or validation error
          500:
            description: Internal server error
        """
        try:
            # Get form data
            object_name = request.form.get('object_name')
            description = request.form.get('description', '')
            
            if not object_name:
                return create_error_response(ValidationAPIError("object_name is required"))
            
            # Validate object name
            object_name = object_name.strip().lower()
            if len(object_name) < 2 or len(object_name) > 50:
                return create_error_response(ValidationAPIError("object_name must be 2-50 characters"))
            
            # Get support images
            support_images = request.files.getlist('support_images')
            if not support_images or len(support_images) == 0:
                return create_error_response(ValidationAPIError("At least one support image is required"))
            
            if len(support_images) > 10:
                return create_error_response(ValidationAPIError("Maximum 10 support images allowed"))
            
            # Validate and process support images
            processed_images = []
            for i, file in enumerate(support_images):
                try:
                    # Validate file
                    if not file or file.filename == '':
                        continue
                    
                    # Read image
                    file_bytes = file.read()
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        return create_error_response(ValidationAPIError(f"Invalid image file: {file.filename}"))
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Validate image size
                    height, width = image.shape[:2]
                    if width < 32 or height < 32:
                        return create_error_response(ValidationAPIError(f"Image too small: {file.filename}"))
                    if width > 2048 or height > 2048:
                        return create_error_response(ValidationAPIError(f"Image too large: {file.filename}"))
                    
                    processed_images.append(image)
                    
                except Exception as e:
                    return create_error_response(ValidationAPIError(f"Error processing image {file.filename}: {str(e)}"))
            
            if len(processed_images) < 1:
                return create_error_response(ValidationAPIError("No valid support images provided"))
            
            # Register object type using few-shot learning
            result = fewshot_service.register_object_type(
                object_name=object_name,
                support_images=processed_images,
                description=description
            )
            
            if result['success']:
                record_api_request('/api/fewshot/register', 'POST', 201)
                return make_response(jsonify(result), 201)
            else:
                record_api_request('/api/fewshot/register', 'POST', 400)
                return create_error_response(ValidationAPIError(result['error']))
                
        except ValidationAPIError as e:
            record_api_request('/api/fewshot/register', 'POST', 400)
            return create_error_response(e)
        except Exception as e:
            logger.error(f"Unexpected error in few-shot registration: {e}")
            record_api_request('/api/fewshot/register', 'POST', 500)
            return create_error_response(ValidationAPIError("Internal server error"))


class FewShotCount(Resource):
    """
    Count objects using few-shot learning
    
    POST /api/fewshot/count
    - Count objects in an image using few-shot classification
    """
    
    @track_response_time('api_fewshot_count')
    def post(self):
        """
        Count objects using few-shot learning
        ---
        tags:
          - Few-Shot Learning
        parameters:
          - in: formData
            name: image
            type: file
            required: true
            description: Image file to process
          - in: formData
            name: object_name
            type: string
            required: true
            description: Name of the few-shot object type
          - in: formData
            name: description
            type: string
            required: false
            description: Optional description
        responses:
          200:
            description: Objects counted successfully
            schema:
              type: object
              properties:
                success:
                  type: boolean
                predicted_count:
                  type: integer
                confidence:
                  type: number
                distance_to_prototype:
                  type: number
                object_type:
                  type: string
                processing_time_ms:
                  type: number
                feature_extraction_time_ms:
                  type: number
                classification_time_ms:
                  type: number
                prediction_id:
                  type: string
          400:
            description: Bad request or validation error
          404:
            description: Few-shot object type not found
          500:
            description: Internal server error
        """
        try:
            # Validate file upload
            try:
                file = validate_file_upload(request)
            except ValidationAPIError as e:
                record_api_request('/api/fewshot/count', 'POST', 400)
                return create_error_response(e)
            
            # Get form data
            object_name = request.form.get('object_name')
            description = request.form.get('description', f'Count {object_name} objects using few-shot learning')
            
            if not object_name:
                record_api_request('/api/fewshot/count', 'POST', 400)
                return create_error_response(ValidationAPIError("object_name is required"))
            
            object_name = object_name.strip().lower()
            
            # Process image
            try:
                # Read image
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    record_api_request('/api/fewshot/count', 'POST', 400)
                    return create_error_response(ValidationAPIError("Invalid image file"))
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            except Exception as e:
                record_api_request('/api/fewshot/count', 'POST', 400)
                return create_error_response(ValidationAPIError(f"Error processing image: {str(e)}"))
            
            # Count objects using few-shot learning
            result = fewshot_service.count_objects_fewshot(
                image=image,
                object_name=object_name
            )
            
            if result['success']:
                record_api_request('/api/fewshot/count', 'POST', 200)
                return make_response(jsonify(result), 200)
            else:
                if "not found" in result['error'].lower():
                    record_api_request('/api/fewshot/count', 'POST', 404)
                    return create_error_response(ValidationAPIError(result['error']), 404)
                else:
                    record_api_request('/api/fewshot/count', 'POST', 400)
                    return create_error_response(ValidationAPIError(result['error']))
                
        except ValidationAPIError as e:
            record_api_request('/api/fewshot/count', 'POST', 400)
            return create_error_response(e)
        except Exception as e:
            logger.error(f"Unexpected error in few-shot counting: {e}")
            record_api_request('/api/fewshot/count', 'POST', 500)
            return create_error_response(ValidationAPIError(f"Internal server error: {str(e)}"))


class FewShotObjectTypes(Resource):
    """
    Manage few-shot object types
    
    GET /api/fewshot/object-types - List all few-shot object types
    DELETE /api/fewshot/object-types/<object_name> - Delete a few-shot object type
    """
    
    @track_response_time('api_fewshot_object_types')
    def get(self):
        """
        Get all registered few-shot object types
        ---
        tags:
          - Few-Shot Learning
        responses:
          200:
            description: List of few-shot object types
            schema:
              type: object
              properties:
                success:
                  type: boolean
                object_types:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: string
                      name:
                        type: string
                      description:
                        type: string
                      support_images_count:
                        type: integer
                      is_active:
                        type: boolean
                      created_at:
                        type: string
                      updated_at:
                        type: string
        """
        try:
            object_types = fewshot_service.get_fewshot_object_types()
            
            result = {
                'success': True,
                'object_types': object_types,
                'total_count': len(object_types)
            }
            
            record_api_request('/api/fewshot/object-types', 'GET', 200)
            return make_response(jsonify(result), 200)
            
        except Exception as e:
            logger.error(f"Error getting few-shot object types: {e}")
            record_api_request('/api/fewshot/object-types', 'GET', 500)
            return create_error_response(ValidationAPIError("Internal server error"))


class FewShotObjectTypeSingle(Resource):
    """
    Manage individual few-shot object types
    
    GET /api/fewshot/object-types/<object_name> - Get specific few-shot object type
    DELETE /api/fewshot/object-types/<object_name> - Delete specific few-shot object type
    """
    
    @track_response_time('api_fewshot_object_type_single')
    def get(self, object_name):
        """
        Get specific few-shot object type
        ---
        tags:
          - Few-Shot Learning
        parameters:
          - in: path
            name: object_name
            type: string
            required: true
            description: Name of the few-shot object type
        responses:
          200:
            description: Few-shot object type details
          404:
            description: Object type not found
        """
        try:
            object_type = fewshot_service.get_fewshot_object_type(object_name)
            
            if object_type:
                record_api_request(f'/api/fewshot/object-types/{object_name}', 'GET', 200)
                return make_response(jsonify({'success': True, 'object_type': object_type}), 200)
            else:
                record_api_request(f'/api/fewshot/object-types/{object_name}', 'GET', 404)
                return create_error_response(ValidationAPIError(f"Few-shot object type '{object_name}' not found"), 404)
                
        except Exception as e:
            logger.error(f"Error getting few-shot object type {object_name}: {e}")
            record_api_request(f'/api/fewshot/object-types/{object_name}', 'GET', 500)
            return create_error_response(ValidationAPIError("Internal server error"))
    
    @track_response_time('api_fewshot_object_type_delete')
    def delete(self, object_name):
        """
        Delete specific few-shot object type
        ---
        tags:
          - Few-Shot Learning
        parameters:
          - in: path
            name: object_name
            type: string
            required: true
            description: Name of the few-shot object type to delete
        responses:
          200:
            description: Object type deleted successfully
          404:
            description: Object type not found
        """
        try:
            result = fewshot_service.delete_fewshot_object_type(object_name)
            
            if result['success']:
                record_api_request(f'/api/fewshot/object-types/{object_name}', 'DELETE', 200)
                return make_response(jsonify(result), 200)
            else:
                if "not found" in result['error'].lower():
                    record_api_request(f'/api/fewshot/object-types/{object_name}', 'DELETE', 404)
                    return create_error_response(ValidationAPIError(result['error']), 404)
                else:
                    record_api_request(f'/api/fewshot/object-types/{object_name}', 'DELETE', 400)
                    return create_error_response(ValidationAPIError(result['error']))
                    
        except Exception as e:
            logger.error(f"Error deleting few-shot object type {object_name}: {e}")
            record_api_request(f'/api/fewshot/object-types/{object_name}', 'DELETE', 500)
            return create_error_response(ValidationAPIError("Internal server error"))


class FewShotPredictions(Resource):
    """
    Get few-shot learning predictions for monitoring and analysis
    
    GET /api/fewshot/predictions - List recent few-shot predictions
    """
    
    @track_response_time('api_fewshot_predictions')
    def get(self):
        """
        Get recent few-shot learning predictions
        ---
        tags:
          - Few-Shot Learning
        parameters:
          - in: query
            name: object_name
            type: string
            required: false
            description: Filter by object type name
          - in: query
            name: limit
            type: integer
            required: false
            description: Maximum number of predictions to return (default: 50)
        responses:
          200:
            description: List of recent predictions
        """
        try:
            # Get query parameters
            object_name = request.args.get('object_name')
            limit = int(request.args.get('limit', 50))
            
            # Build query
            query = database.session.query(FewShotPrediction)
            
            if object_name:
                # Join with FewShotObjectType to filter by name
                query = query.join(FewShotObjectType).filter(FewShotObjectType.name == object_name)
            
            # Order by creation time (newest first) and limit
            predictions = query.order_by(FewShotPrediction.created_at.desc()).limit(limit).all()
            
            # Convert to dictionaries
            predictions_data = [pred.to_dict() for pred in predictions]
            
            result = {
                'success': True,
                'predictions': predictions_data,
                'total_count': len(predictions_data),
                'filters': {
                    'object_name': object_name,
                    'limit': limit
                }
            }
            
            record_api_request('/api/fewshot/predictions', 'GET', 200)
            return make_response(jsonify(result), 200)
            
        except Exception as e:
            logger.error(f"Error getting few-shot predictions: {e}")
            record_api_request('/api/fewshot/predictions', 'GET', 500)
            return create_error_response(ValidationAPIError("Internal server error"))

