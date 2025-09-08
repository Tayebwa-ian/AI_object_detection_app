#!/usr/bin/python3
"""
Enhanced Error Handling Utilities
"""
from flask import jsonify, make_response
from marshmallow import ValidationError
import traceback
import logging

# Configure logging
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom API exception with status code and user-friendly message"""
    def __init__(self, message, status_code=500, error_code=None, details=None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)

class ValidationAPIError(APIError):
    """Validation error with 400 status code"""
    def __init__(self, message, details=None):
        super().__init__(message, 400, 'VALIDATION_ERROR', details)

class NotFoundAPIError(APIError):
    """Not found error with 404 status code"""
    def __init__(self, message, details=None):
        super().__init__(message, 404, 'NOT_FOUND', details)

class ProcessingAPIError(APIError):
    """Processing error with 422 status code"""
    def __init__(self, message, details=None):
        super().__init__(message, 422, 'PROCESSING_ERROR', details)

class DatabaseAPIError(APIError):
    """Database error with 500 status code"""
    def __init__(self, message, details=None):
        super().__init__(message, 500, 'DATABASE_ERROR', details)

def create_error_response(error, include_details=False):
    """Create standardized error response"""
    if isinstance(error, APIError):
        response_data = {
            'success': False,
            'error': {
                'message': error.message,
                'code': error.error_code,
                'status_code': error.status_code
            }
        }
        if include_details and error.details:
            response_data['error']['details'] = error.details
        
        return make_response(jsonify(response_data), error.status_code)
    
    elif isinstance(error, ValidationError):
        response_data = {
            'success': False,
            'error': {
                'message': 'Validation failed',
                'code': 'VALIDATION_ERROR',
                'status_code': 400,
                'details': error.messages
            }
        }
        return make_response(jsonify(response_data), 400)
    
    else:
        # Generic error
        logger.error(f"Unhandled error: {str(error)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        response_data = {
            'success': False,
            'error': {
                'message': 'An unexpected error occurred',
                'code': 'INTERNAL_ERROR',
                'status_code': 500
            }
        }
        
        if include_details:
            response_data['error']['details'] = str(error)
        
        return make_response(jsonify(response_data), 500)

def handle_file_upload_error(error):
    """Handle file upload specific errors"""
    if 'No image file provided' in str(error):
        return create_error_response(
            ValidationAPIError(
                'No image file was uploaded',
                'Please select an image file to process'
            )
        )
    elif 'Invalid file type' in str(error):
        return create_error_response(
            ValidationAPIError(
                'Invalid file type',
                'Please upload a valid image file (JPG, PNG, BMP)'
            )
        )
    elif 'File too large' in str(error):
        return create_error_response(
            ValidationAPIError(
                'File too large',
                'Please upload an image smaller than 10MB'
            )
        )
    else:
        return create_error_response(
            ProcessingAPIError(
                'Failed to upload image',
                str(error)
            )
        )

def handle_ai_processing_error(error):
    """Handle AI processing specific errors"""
    if 'models not loaded' in str(error).lower():
        return create_error_response(
            ProcessingAPIError(
                'AI models are not available',
                'The AI processing system is currently unavailable. Please try again later.'
            )
        )
    elif 'memory' in str(error).lower() or 'cuda' in str(error).lower():
        return create_error_response(
            ProcessingAPIError(
                'Insufficient memory for processing',
                'The image is too large or complex for processing. Try a smaller image.'
            )
        )
    elif 'timeout' in str(error).lower():
        return create_error_response(
            ProcessingAPIError(
                'Processing timeout',
                'The image took too long to process. Try a smaller or simpler image.'
            )
        )
    else:
        return create_error_response(
            ProcessingAPIError(
                'AI processing failed',
                str(error)
            )
        )

def handle_database_error(error):
    """Handle database specific errors"""
    if 'not found' in str(error).lower():
        return create_error_response(
            NotFoundAPIError(
                'Record not found',
                'The requested data could not be found in the database'
            )
        )
    elif 'duplicate' in str(error).lower() or 'unique' in str(error).lower():
        return create_error_response(
            ValidationAPIError(
                'Duplicate entry',
                'A record with this information already exists'
            )
        )
    else:
        return create_error_response(
            DatabaseAPIError(
                'Database operation failed',
                'There was an error accessing the database'
            )
        )

def validate_required_fields(data, required_fields):
    """Validate that required fields are present"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationAPIError(
            f'Missing required fields: {", ".join(missing_fields)}',
            f'Please provide the following required fields: {", ".join(missing_fields)}'
        )

def validate_file_upload(request):
    """Validate file upload request"""
    if 'image' not in request.files:
        raise ValidationAPIError(
            'No image file provided',
            'Please select an image file to upload'
        )
    
    file = request.files['image']
    if file.filename == '':
        raise ValidationAPIError(
            'No file selected',
            'Please select a valid image file'
        )
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        raise ValidationAPIError(
            'Invalid file type',
            'Please upload a valid image file (PNG, JPG, JPEG, BMP, GIF)'
        )
    
    # Check file size (10MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise ValidationAPIError(
            'File too large',
            'Please upload an image smaller than 10MB'
        )
    
    return file

def validate_object_type(object_type):
    """Validate object type parameter"""
    if not object_type:
        raise ValidationAPIError(
            'Object type is required',
            'Please specify the type of object to count'
        )
    
    if len(object_type.strip()) < 2:
        raise ValidationAPIError(
            'Object type too short',
            'Object type must be at least 2 characters long'
        )
    
    if len(object_type.strip()) > 50:
        raise ValidationAPIError(
            'Object type too long',
            'Object type must be less than 50 characters'
        )
    
    return object_type.strip()








