#!/usr/bin/python3
"""
Input Views module
"""
from flask_restful import Resource
from ...storage import database, Input, Output, ObjectType
from ..serializers.inputs import InputSchema
from marshmallow import ValidationError, EXCLUDE
from flask import request, jsonify, make_response
from ..utils.image_utils import upload_image
from ...config import config
from ...pipeline.pipeline import pipeline
from .monitoring import monitoring
from ..utils.error_handlers import (
    create_error_response, handle_file_upload_error, handle_ai_processing_error,
    handle_database_error, validate_file_upload, validate_object_type,
    ValidationAPIError, ProcessingAPIError, DatabaseAPIError
)
import os
import uuid


input_schema = InputSchema(unknown=EXCLUDE)
inputs_schema = InputSchema(many=True)


class InputList(Resource):
    """Handles requests for multiple Inputs"""

    def get(self):
        """
        Get all inputs
        ---
        tags:
          - Inputs
        responses:
          200:
            description: List of all inputs
            schema:
              type: array
              items:
                $ref: '#/definitions/Input'
          400:
            description: Could not fetch data from storage
        """
        inputs = database.all(Input)
        if not inputs:
            response = {
                "status": "error",
                "message": "could not fetch data from the storage",
                "data": inputs
            }
            return make_response(jsonify(response), 400)
        return inputs_schema.dump(inputs), 200

    def post(self):
        """
        Upload image and process with AI pipeline
        ---
        tags:
          - Inputs
        parameters:
          - in: formData
            name: image
            type: file
            required: true
            description: Image file to process
          - in: formData
            name: object_type
            type: string
            required: true
            description: Type of object to count
          - in: formData
            name: description
            type: string
            required: false
            description: Optional description
        responses:
          201:
            description: Image processed successfully
            schema:
              type: object
              properties:
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
                image_path:
                  type: string
                created_at:
                  type: string
          400:
            description: Bad request or processing error
          500:
            description: Internal server error
        """
        try:
            # Validate file upload
            try:
                file = validate_file_upload(request)
            except ValidationAPIError as e:
                return create_error_response(e)
            
            # Get and validate form data
            object_type = request.form.get('object_type')
            description = request.form.get('description', f'Count {object_type} objects')
            
            try:
                object_type = validate_object_type(object_type)
            except ValidationAPIError as e:
                return create_error_response(e)
            
            # Upload image
            try:
                image_result = upload_image(request)
                if isinstance(image_result, tuple):  # Error response
                    return handle_file_upload_error(image_result[0].get_json().get('error', 'Upload failed'))
            except Exception as e:
                return handle_file_upload_error(e)
            
            image_filename = image_result
            # Logical path for frontend/API
            image_path = os.path.join('media', image_filename)
            # Filesystem path for pipeline processing
            fs_image_path = os.path.join(config.MEDIA_DIRECTORY, image_filename)
            
            # Process image with AI pipeline
            print(f"Processing image: {image_path} for object type: {object_type}")
            try:
                ai_result = pipeline.process_image(fs_image_path, object_type)
                
                if not ai_result.get('success', False):
                    return handle_ai_processing_error(
                        Exception(ai_result.get("error", "Unknown AI processing error"))
                    )
            except Exception as e:
                return handle_ai_processing_error(e)
            
            # Create input record
            input_data = {
                'description': description,
                'image_path': image_path
            }
            new_input = Input(**input_data)
            new_input.save()
            
            # Get or create object type
            object_type_record = database.get(ObjectType, name=object_type)
            if not object_type_record:
                object_type_record = ObjectType(
                    name=object_type,
                    description=f'Object type for {object_type}'
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
            
            # Prepare response
            response_data = {
                'success': True,
                'result_id': str(new_output.id),
                'object_type': object_type,
                'predicted_count': ai_result.get('predicted_count', 0),
                'confidence': ai_result.get('confidence', 0.0),
                'processing_time': ai_result.get('processing_time', 0.0),
                'image_path': image_path,
                'created_at': new_output.created_at.isoformat() if hasattr(new_output, 'created_at') else None
            }
            
            # Record performance metrics
            processing_time = ai_result.get('processing_time', 0.0)
            success = ai_result.get('success', True)
            monitoring.record_request(object_type, processing_time, success)
            
            print(f"Successfully processed image: {ai_result.get('predicted_count', 0)} {object_type}s detected")
            return make_response(jsonify(response_data), 201)
            
        except Exception as e:
            # Record failed request
            monitoring.record_request(object_type, 0.0, False)
            print(f"Error processing image: {str(e)}")
            return create_error_response(e, include_details=True)

    def count_all_objects(self):
        """
        Upload image and detect all objects (auto-detection)
        ---
        tags:
          - Inputs
        parameters:
          - in: formData
            name: image
            type: file
            required: true
            description: Image file to process
          - in: formData
            name: object_type
            type: string
            required: true
            description: Primary object type to focus on
          - in: formData
            name: description
            type: string
            required: false
            description: Optional description
        responses:
          201:
            description: Image processed successfully with all objects detected
            schema:
              type: object
              properties:
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
                image_path:
                  type: string
                created_at:
                  type: string
          400:
            description: Bad request or processing error
          500:
            description: Internal server error
        """
        try:
            # Check if image file is present
            if 'image' not in request.files:
                return make_response(jsonify({
                    'error': 'No image file provided'
                }), 400)
            
            # Get form data
            object_type = request.form.get('object_type')
            description = request.form.get('description', f'Detect and count all objects in this image')
            
            if not object_type:
                return make_response(jsonify({
                    'error': 'object_type is required'
                }), 400)
            
            # Upload image
            image_result = upload_image(request)
            if isinstance(image_result, tuple):  # Error response
                return image_result
            
            image_filename = image_result
            image_path = os.path.join('media', image_filename)
            fs_image_path = os.path.join(config.MEDIA_DIRECTORY, image_filename)
            
            # Process image with AI pipeline (auto-detection)
            print(f"Auto-detecting objects in image: {image_path}")
            ai_result = pipeline.process_image_auto(fs_image_path)
            
            if not ai_result.get('success', False):
                return make_response(jsonify({
                    'error': f'AI processing failed: {ai_result.get("error", "Unknown error")}'
                }), 500)
            
            # Create input record
            input_data = {
                'description': description,
                'image_path': image_path
            }
            new_input = Input(**input_data)
            new_input.save()
            
            # Get or create object type
            object_type_record = database.get(ObjectType, name=object_type)
            if not object_type_record:
                object_type_record = ObjectType(
                    name=object_type,
                    description=f'Object type for {object_type}'
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
            
            # Prepare response
            response_data = {
                'success': True,
                'result_id': str(new_output.id),
                'object_type': object_type,
                'predicted_count': ai_result.get('predicted_count', 0),
                'confidence': ai_result.get('confidence', 0.0),
                'processing_time': ai_result.get('processing_time', 0.0),
                'image_path': image_path,
                'created_at': new_output.created_at.isoformat() if hasattr(new_output, 'created_at') else None
            }
            
            # Record performance metrics
            processing_time = ai_result.get('processing_time', 0.0)
            success = ai_result.get('success', True)
            monitoring.record_request(f"{object_type}_auto", processing_time, success)
            
            print(f"Successfully auto-detected objects: {ai_result.get('predicted_count', 0)} total objects")
            return make_response(jsonify(response_data), 201)
            
        except Exception as e:
            # Record failed request
            monitoring.record_request(f"{object_type}_auto", 0.0, False)
            print(f"Error auto-detecting objects: {str(e)}")
            return create_error_response(e, include_details=True)


class InputSingle(Resource):
    """Handles operations on a single Input"""

    def get(self, input_id):
        """
        Get a single input
        ---
        tags:
          - Inputs
        parameters:
          - in: path
            name: input_id
            type: integer
            required: true
            description: ID of the Input to retrieve
        responses:
          200:
            description: Input retrieved successfully
            schema:
              $ref: '#/definitions/Input'
          404:
            description: Input not found
        """
        input = database.get(Input, id=input_id)
        if input:
            return (input_schema.dump(input), 200)

    def delete(self, input_id):
        """
        Delete an input
        ---
        tags:
          - Inputs
        parameters:
          - in: path
            name: input_id
            type: integer
            required: true
            description: ID of the Input to delete
        responses:
          200:
            description: Input successfully deleted
          404:
            description: Input not found
        """
        input = database.get(Input, id=input_id)
        database.delete(input)
        response = {'message': 'resource successfully deleted'}
        return make_response(jsonify(response), 200)

    def put(self, input_id):
        """
        Update an input
        ---
        tags:
          - Inputs
        parameters:
          - in: path
            name: input_id
            type: integer
            required: true
            description: ID of the Input to update
          - in: body
            name: body
            required: true
            schema:
              $ref: '#/definitions/Input'
        responses:
          200:
            description: Input updated successfully
            schema:
              $ref: '#/definitions/Input'
          403:
            description: Validation error
        """
        data = request.get_json()
        try:
            data = input_schema.load(data)
        except ValidationError as e:
            responseobject = {
                "status": "fail",
                "message": e.messages
            }
            return make_response(jsonify(responseobject), 403)
        input = database.update(Input, input_id, **data)
        return input_schema.dump(input), 200
