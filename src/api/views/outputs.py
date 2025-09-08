#!/usr/bin/python3
"""
Output Views module
"""
from flask_restful import Resource
from ...storage import database, Output, ObjectType, Input
from ..serializers.outputs import OutputSchema
from marshmallow import ValidationError, EXCLUDE
from flask import request, jsonify, make_response
from ..utils.error_handlers import (
    create_error_response, handle_database_error, NotFoundAPIError
)


output_schema = OutputSchema(unknown=EXCLUDE)
outputs_schema = OutputSchema(many=True)


class OutputList(Resource):
    """Handles multiple Outputs"""

    def get(self):
        """
        Get all outputs
        ---
        tags:
          - Outputs
        summary: Retrieve all output records
        description: Returns all stored outputs with metadata.
        responses:
          200:
            description: A list of outputs
            schema:
              type: array
              items:
                $ref: '#/definitions/Output'
          400:
            description: Could not fetch data from storage
        """
        try:
            from ...storage import ObjectType, Input
            
            # Handle uninitialized database gracefully
            if database is None or not hasattr(database, 'all'):
                return [], 200
            
            # Pagination and filtering params
            try:
                page = int(request.args.get('page', 1))
                per_page = int(request.args.get('per_page', 20))
            except Exception:
                page, per_page = 1, 20
            per_page = max(1, min(per_page, 100))
            filter_object_type = request.args.get('object_type')

            outputs = database.all(Output) or []

            # Enhance outputs with object type names and image paths
            enhanced_outputs = []
            for output in outputs:
                # Get object type name
                object_type = database.get(ObjectType, id=output.object_type_id)
                object_type_name = object_type.name if object_type else "Unknown"
                
                # Get image path
                input_record = database.get(Input, id=output.input_id)
                image_path = input_record.image_path if input_record else "Unknown"
                
                # Create enhanced output
                enhanced_output = {
                    'id': output.id,
                    'created_at': output.created_at.isoformat() if hasattr(output, 'created_at') else None,
                    'updated_at': output.updated_at.isoformat() if hasattr(output, 'updated_at') else None,
                    'predicted_count': output.predicted_count,
                    'corrected_count': output.corrected_count,
                    'pred_confidence': output.pred_confidence,
                    'object_type_id': output.object_type_id,
                    'input_id': output.input_id,
                    'object_type': object_type_name,
                    'image_path': image_path
                }
                enhanced_outputs.append(enhanced_output)

            # Optional filter by object type name
            if filter_object_type:
                filter_l = filter_object_type.strip().lower()
                enhanced_outputs = [e for e in enhanced_outputs if (e.get('object_type') or '').lower() == filter_l]

            # Server-side pagination
            total = len(enhanced_outputs)
            start = max(0, (page - 1) * per_page)
            end = start + per_page
            paged = enhanced_outputs[start:end]

            resp = make_response(jsonify(paged), 200)
            resp.headers['X-Total-Count'] = str(total)
            resp.headers['X-Page'] = str(page)
            resp.headers['X-Per-Page'] = str(per_page)
            return resp
        except Exception as e:
            # Return an empty list instead of 500 to keep history page functional,
            # but include error message for debugging
            return make_response(jsonify({
                'error': f'Failed to fetch results: {str(e)}',
                'results': []
            }), 200)

    def post(self):
        """
        Create a new output
        ---
        tags:
          - Outputs
        summary: Submit a new output record
        description: |
          Accepts prediction results for a given input, including predicted count,
          optional corrected count, confidence level, and foreign keys to object type and input.
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required:
                - predicted_count
                - pred_confidence
                - object_type_id
                - input_id
              properties:
                predicted_count:
                  type: integer
                  example: 5
                corrected_count:
                  type: integer
                  example: 4
                pred_confidence:
                  type: number
                  format: float
                  example: 0.95
                object_type_id:
                  type: string
                  example: "object-type-uuid"
                input_id:
                  type: string
                  example: "input-uuid"
        responses:
          201:
            description: Output created successfully
            schema:
              $ref: '#/definitions/Output'
          403:
            description: Validation error
        """
        data = request.get_json()
        try:
            data = output_schema.load(data)
        except ValidationError as e:
            responseobject = {
                "status": "fail",
                "message": e.messages
            }
            return make_response(jsonify(responseobject), 403)

        new_output = Output(**data)
        new_output.save()
        return output_schema.dump(new_output), 201


class OutputSingle(Resource):
    """Handles single Output operations"""

    def get(self, output_id):
        """
        Get a single output
        ---
        tags:
          - Outputs
        summary: Retrieve an output by ID
        parameters:
          - in: path
            name: output_id
            type: string
            required: true
            description: UUID of the output
        responses:
          200:
            description: Output retrieved successfully
            schema:
              $ref: '#/definitions/Output'
          404:
            description: Output not found
        """
        try:
            output = database.get(Output, id=output_id)
            if not output:
                return create_error_response(
                    NotFoundAPIError(
                        f'Output with ID {output_id} not found',
                        'The requested output record does not exist'
                    )
                )

            # Enhance with object type name and image path for frontend details view
            object_type = database.get(ObjectType, id=output.object_type_id)
            input_record = database.get(Input, id=output.input_id)

            enhanced = output_schema.dump(output)
            enhanced['object_type'] = object_type.name if object_type else 'Unknown'
            enhanced['image_path'] = input_record.image_path if input_record else None

            return (enhanced, 200)
        except Exception as e:
            return handle_database_error(e)

    def delete(self, output_id):
        """
        Delete an output
        ---
        tags:
          - Outputs
        summary: Delete an output by ID
        parameters:
          - in: path
            name: output_id
            type: string
            required: true
            description: UUID of the output to delete
        responses:
          200:
            description: Output successfully deleted
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "resource successfully deleted"
          404:
            description: Output not found
        """
        output = database.get(Output, id=output_id)
        database.delete(output)
        response = {'message': 'resource successfully deleted'}
        return make_response(jsonify(response), 200)

    def put(self, output_id):
        """
        Update an output (correction endpoint)
        ---
        tags:
          - Outputs
        summary: Submit a correction for a prediction
        description: Update the corrected count for an existing output prediction.
        parameters:
          - in: path
            name: output_id
            type: string
            required: true
            description: UUID of the output to correct
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                corrected_count:
                  type: integer
                  example: 5
                  description: The corrected count provided by the user
        responses:
          200:
            description: Correction submitted successfully
            schema:
              type: object
              properties:
                success:
                  type: boolean
                result_id:
                  type: string
                predicted_count:
                  type: integer
                corrected_count:
                  type: integer
                updated_at:
                  type: string
                message:
                  type: string
          403:
            description: Validation error
          404:
            description: Output not found
        """
        try:
            # Get the existing output
            output = database.get(Output, id=output_id)
            if not output:
                return create_error_response(
                    NotFoundAPIError(
                        f'Output with ID {output_id} not found',
                        'The requested output record does not exist'
                    )
                )
            
            # Get the correction data
            data = request.get_json()
            if not data or 'corrected_count' not in data:
                return make_response(jsonify({
                    'error': 'corrected_count is required'
                }), 400)
            
            corrected_count = data['corrected_count']
            
            # Update only the corrected count
            output.corrected_count = corrected_count
            output.save()
            
            # Prepare response
            response_data = {
                'success': True,
                'result_id': str(output.id),
                'predicted_count': output.predicted_count,
                'corrected_count': output.corrected_count,
                'updated_at': output.updated_at.isoformat() if hasattr(output, 'updated_at') else None,
                'message': 'Correction submitted successfully'
            }
            
            return make_response(jsonify(response_data), 200)
            
        except Exception as e:
            return handle_database_error(e)
