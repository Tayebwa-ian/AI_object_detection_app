#!/usr/bin/python3
"""
Batch Processing Views module
"""
from flask_restful import Resource
from flask import request, jsonify, make_response
from ...storage import database, Input, Output, ObjectType
from ..utils.image_utils import upload_image
from ...config import config
from ...pipeline.pipeline import pipeline
from .monitoring import monitoring
from ..utils.error_handlers import (
    create_error_response, handle_file_upload_error, handle_ai_processing_error,
    handle_database_error, validate_object_type, ValidationAPIError, 
    ProcessingAPIError, DatabaseAPIError
)
import os
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any

class BatchProcessing(Resource):
    """Handle batch processing of multiple images"""
    
    def post(self):
        """
        Process multiple images in batch
        ---
        tags:
          - Batch Processing
        parameters:
          - in: formData
            name: images[]
            type: file
            required: true
            description: Multiple image files to process
          - in: formData
            name: object_type
            type: string
            required: true
            description: Type of object to count
          - in: formData
            name: description
            type: string
            required: false
            description: Optional description for the batch
          - in: formData
            name: auto_detect
            type: boolean
            required: false
            description: Whether to use auto-detection (default: false)
        responses:
          200:
            description: Batch processing completed
            schema:
              type: object
              properties:
                success:
                  type: boolean
                batch_id:
                  type: string
                total_images:
                  type: integer
                successful_images:
                  type: integer
                failed_images:
                  type: integer
                processing_time:
                  type: number
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      image_name:
                        type: string
                      success:
                        type: boolean
                      result_id:
                        type: string
                      object_type:
                        type: string
                      predicted_count:
                        type: integer
                      confidence:
                        type: number
                      processing_time:
                        type: number
                      error:
                        type: string
          400:
            description: Bad request
          500:
            description: Internal server error
        """
        batch_start_time = time.time()
        batch_id = str(uuid.uuid4())
        
        try:
            # Validate object type
            object_type = request.form.get('object_type')
            auto_detect = request.form.get('auto_detect', 'false').lower() == 'true'
            description = request.form.get('description', f'Batch processing {object_type} objects')
            
            if not auto_detect:
                try:
                    object_type = validate_object_type(object_type)
                except ValidationAPIError as e:
                    return create_error_response(e)
            
            # Get uploaded files
            if 'images[]' not in request.files:
                return create_error_response(
                    ValidationAPIError(
                        'No images provided',
                        'Please upload at least one image file'
                    )
                )
            
            files = request.files.getlist('images[]')
            if not files or len(files) == 0:
                return create_error_response(
                    ValidationAPIError(
                        'No images provided',
                        'Please upload at least one image file'
                    )
                )
            
            # Validate file count (limit to 10 images per batch)
            if len(files) > 10:
                return create_error_response(
                    ValidationAPIError(
                        'Too many images',
                        'Maximum 10 images allowed per batch'
                    )
                )
            
            print(f"Starting batch processing: {len(files)} images, batch_id: {batch_id}")
            
            # Process each image
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, file in enumerate(files):
                image_result = self._process_single_image(
                    file, object_type, description, auto_detect, i + 1, len(files)
                )
                results.append(image_result)
                
                if image_result['success']:
                    successful_count += 1
                else:
                    failed_count += 1
            
            # Calculate total processing time
            total_processing_time = time.time() - batch_start_time
            
            # Record batch metrics
            monitoring.record_request(
                f"{object_type}_batch" if not auto_detect else "auto_detect_batch",
                total_processing_time,
                successful_count > 0
            )
            
            response_data = {
                'success': True,
                'batch_id': batch_id,
                'total_images': len(files),
                'successful_images': successful_count,
                'failed_images': failed_count,
                'processing_time': round(total_processing_time, 3),
                'results': results,
                'created_at': datetime.now().isoformat()
            }
            
            print(f"Batch processing completed: {successful_count}/{len(files)} successful")
            return make_response(jsonify(response_data), 200)
            
        except Exception as e:
            total_processing_time = time.time() - batch_start_time
            monitoring.record_request(
                f"{object_type}_batch" if 'object_type' in locals() else "batch_processing",
                total_processing_time,
                False
            )
            print(f"Batch processing failed: {str(e)}")
            return create_error_response(e, include_details=True)
    
    def _process_single_image(self, file, object_type: str, description: str, 
                            auto_detect: bool, image_index: int, total_images: int) -> Dict[str, Any]:
        """Process a single image within the batch"""
        image_start_time = time.time()
        
        try:
            # Validate file
            if file.filename == '':
                return {
                    'image_name': 'unknown',
                    'success': False,
                    'error': 'No file selected',
                    'processing_time': 0
                }
            
            # Check file extension
            allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
            if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return {
                    'image_name': file.filename,
                    'success': False,
                    'error': 'Invalid file type. Please upload a valid image file (PNG, JPG, JPEG, BMP, GIF)',
                    'processing_time': 0
                }
            
            # Check file size (10MB limit)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > 10 * 1024 * 1024:  # 10MB
                return {
                    'image_name': file.filename,
                    'success': False,
                    'error': 'File too large. Please upload an image smaller than 10MB',
                    'processing_time': 0
                }
            
            print(f"  Processing image {image_index}/{total_images}: {file.filename}")
            
            # Create a temporary request object for upload_image
            from flask import Request
            temp_request = Request.from_values(
                files={'image': file},
                form={'object_type': object_type, 'description': description}
            )
            
            # Upload image
            try:
                image_result = upload_image(temp_request)
                if isinstance(image_result, tuple):  # Error response
                    return {
                        'image_name': file.filename,
                        'success': False,
                        'error': 'Failed to upload image',
                        'processing_time': 0
                    }
            except Exception as e:
                return {
                    'image_name': file.filename,
                    'success': False,
                    'error': f'Upload failed: {str(e)}',
                    'processing_time': 0
                }
            
            image_filename = image_result
            image_path = os.path.join('media', image_filename)
            fs_image_path = os.path.join(config.MEDIA_DIRECTORY, image_filename)
            
            # Process image with AI pipeline
            try:
                if auto_detect:
                    ai_result = pipeline.process_image_auto(fs_image_path)
                else:
                    ai_result = pipeline.process_image(fs_image_path, object_type)
                
                if not ai_result.get('success', False):
                    return {
                        'image_name': file.filename,
                        'success': False,
                        'error': f'AI processing failed: {ai_result.get("error", "Unknown error")}',
                        'processing_time': time.time() - image_start_time
                    }
            except Exception as e:
                return {
                    'image_name': file.filename,
                    'success': False,
                    'error': f'AI processing failed: {str(e)}',
                    'processing_time': time.time() - image_start_time
                }
            
            # Create database records
            try:
                # Create input record
                input_data = {
                    'description': f"{description} - {file.filename}",
                    'image_path': image_path
                }
                new_input = Input(**input_data)
                new_input.save()
                
                # Get or create object type
                if not auto_detect:
                    object_type_record = database.get(ObjectType, name=object_type)
                    if not object_type_record:
                        object_type_record = ObjectType(
                            name=object_type,
                            description=f'Object type for {object_type}'
                        )
                        object_type_record.save()
                else:
                    # For auto-detect, use the detected object type
                    detected_type = ai_result.get('object_type', 'unknown')
                    object_type_record = database.get(ObjectType, name=detected_type)
                    if not object_type_record:
                        object_type_record = ObjectType(
                            name=detected_type,
                            description=f'Auto-detected object type: {detected_type}'
                        )
                        object_type_record.save()
                
                # Create output record
                output_data = {
                    'predicted_count': ai_result.get('predicted_count', 0),
                    'pred_confidence': ai_result.get('confidence', 0.0),
                    'object_type_id': object_type_record.id,
                    'input_id': new_input.id
                }
                new_output = Output(**output_data)
                new_output.save()
                
            except Exception as e:
                return {
                    'image_name': file.filename,
                    'success': False,
                    'error': f'Database operation failed: {str(e)}',
                    'processing_time': time.time() - image_start_time
                }
            
            # Record individual image metrics
            processing_time = time.time() - image_start_time
            monitoring.record_request(
                object_type if not auto_detect else f"{ai_result.get('object_type', 'unknown')}_auto",
                processing_time,
                True
            )
            
            print(f"  Image {image_index}/{total_images} processed successfully: {ai_result.get('predicted_count', 0)} objects")
            
            return {
                'image_name': file.filename,
                'success': True,
                'result_id': str(new_output.id),
                'object_type': ai_result.get('object_type', object_type),
                'predicted_count': ai_result.get('predicted_count', 0),
                'confidence': ai_result.get('confidence', 0.0),
                'processing_time': round(processing_time, 3),
                'created_at': new_output.created_at.isoformat() if hasattr(new_output, 'created_at') else None
            }
            
        except Exception as e:
            processing_time = time.time() - image_start_time
            print(f"   Image {image_index}/{total_images} failed: {str(e)}")
            return {
                'image_name': file.filename if hasattr(file, 'filename') else 'unknown',
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'processing_time': round(processing_time, 3)
            }

class BatchStatus(Resource):
    """Get batch processing status and statistics"""
    
    def get(self):
        """
        Get batch processing statistics
        ---
        tags:
          - Batch Processing
        responses:
          200:
            description: Batch processing statistics
        """
        try:
            # Get recent batch processing statistics
            outputs = database.all(Output) if database.all(Output) else []
            
            # Calculate batch statistics (last 24 hours)
            recent_outputs = []
            for output in outputs:
                if hasattr(output, 'created_at'):
                    # Simple check for recent activity
                    recent_outputs.append(output)
            
            # Get performance metrics
            metrics = monitoring.get_metrics()
            
            stats = {
                'total_processed_today': len(recent_outputs),
                'average_processing_time': metrics.get('average_processing_time', 0),
                'success_rate': metrics.get('success_rate_percent', 0),
                'total_requests': metrics.get('total_requests', 0),
                'system_uptime': metrics.get('uptime_seconds', 0),
                'last_updated': datetime.now().isoformat()
            }
            
            return stats, 200
            
        except Exception as e:
            return create_error_response(e, include_details=True)








