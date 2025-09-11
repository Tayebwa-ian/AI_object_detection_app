#!/usr/bin/python3
"""
Output Views module
"""
from flask_restful import Resource
from ...storage import database, Output
from ..serializers.outputs import OutputSchema
from marshmallow import ValidationError, EXCLUDE
from flask import request, jsonify, make_response


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
        outputs = database.all(Output)
        if not outputs:
            response = {
                "status": "error",
                "message": "could not fetch data from the storage",
                "data": outputs
            }
            return make_response(jsonify(response), 400)
        return outputs_schema.dump(outputs), 200

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
        output = database.get(Output, id=output_id)
        if output:
            return (output_schema.dump(output), 200)

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
        Update an output
        ---
        tags:
          - Outputs
        summary: Update an output record
        description: Modify prediction results or correction for an existing output.
        parameters:
          - in: path
            name: output_id
            type: string
            required: true
            description: UUID of the output to update
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                predicted_count:
                  type: integer
                  example: 6
                corrected_count:
                  type: integer
                  example: 5
                pred_confidence:
                  type: number
                  format: float
                  example: 0.90
                object_type_id:
                  type: string
                  example: "object-type-uuid"
                input_id:
                  type: string
                  example: "input-uuid"
        responses:
          200:
            description: Output updated successfully
            schema:
              $ref: '#/definitions/Output'
          403:
            description: Validation error
          404:
            description: Output not found
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
        output = database.update(Output, output_id, **data)
        return output_schema.dump(output), 200
