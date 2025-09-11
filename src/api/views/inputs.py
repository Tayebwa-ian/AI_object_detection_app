#!/usr/bin/python3
"""
Input Views module
"""
from flask_restful import Resource
from ...storage import database, Input
from ..serializers.inputs import InputSchema
from marshmallow import ValidationError, EXCLUDE
from flask import request, jsonify, make_response


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
        Create a new input
        ---
        tags:
          - Inputs
        parameters:
          - in: body
            name: body
            required: true
            schema:
              $ref: '#/definitions/Input'
        responses:
          201:
            description: Input created successfully
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

        new_input = Input(**data)
        new_input.save()
        return input_schema.dump(new_input), 201


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
