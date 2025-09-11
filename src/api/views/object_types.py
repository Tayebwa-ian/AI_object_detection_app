#!/usr/bin/python3
"""
ObjectType Views module
"""
from flask_restful import Resource
from ...storage import database, ObjectType
from ..serializers.object_types import ObjectTypeSchema
from marshmallow import ValidationError, EXCLUDE
from flask import request, jsonify, make_response


obj_schema = ObjectTypeSchema(unknown=EXCLUDE)
objs_schema = ObjectTypeSchema(many=True)


class ObjectTypeList(Resource):
    """Handles multiple Object Types"""

    def get(self):
        """
        Get all object types
        ---
        tags:
          - Object Types
        summary: Retrieve all object types
        description: Returns a list of all object types in the database.
        responses:
          200:
            description: List of object types
            schema:
              type: array
              items:
                $ref: '#/definitions/ObjectType'
          400:
            description: Could not fetch data from storage
        """
        objs = database.all(ObjectType)
        if not objs:
            response = {
                "status": "error",
                "message": "could not fetch data from the storage",
                "data": objs
            }
            return make_response(jsonify(response), 400)
        return objs_schema.dump(objs), 200

    def post(self):
        """
        Create a new object type
        ---
        tags:
          - Object Types
        summary: Add a new object type
        description: Create a new object type with a unique name and description.
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required:
                - name
                - description
              properties:
                name:
                  type: string
                  example: "Tree"
                description:
                  type: string
                  example: "A tall plant with a trunk and leaves"
        responses:
          201:
            description: Object type created successfully
            schema:
              $ref: '#/definitions/ObjectType'
          403:
            description: Validation error
        """
        data = request.get_json()
        try:
            data = obj_schema.load(data)
        except ValidationError as e:
            responseobject = {
                "status": "fail",
                "message": e.messages
            }
            return make_response(jsonify(responseobject), 403)

        new_obj = ObjectType(**data)
        new_obj.save()
        return obj_schema.dump(new_obj), 201


class ObjectTypeSingle(Resource):
    """Handles single Object Type operations"""

    def get(self, obj_id):
        """
        Get a single object type
        ---
        tags:
          - Object Types
        summary: Retrieve an object type by ID
        parameters:
          - in: path
            name: obj_id
            type: string
            required: true
            description: UUID of the object type
        responses:
          200:
            description: Object type retrieved successfully
            schema:
              $ref: '#/definitions/ObjectType'
          404:
            description: Object type not found
        """
        obj = database.get(ObjectType, id=obj_id)
        if obj:
            return (obj_schema.dump(obj), 200)

    def delete(self, obj_id):
        """
        Delete an object type
        ---
        tags:
          - Object Types
        summary: Delete an object type by ID
        parameters:
          - in: path
            name: obj_id
            type: string
            required: true
            description: UUID of the object type to delete
        responses:
          200:
            description: Object type successfully deleted
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "resource successfully deleted"
          404:
            description: Object type not found
        """
        obj = database.get(ObjectType, id=obj_id)
        database.delete(obj)
        response = {'message': 'resource successfully deleted'}
        return make_response(jsonify(response), 200)

    def put(self, obj_id):
        """
        Update an object type
        ---
        tags:
          - Object Types
        summary: Update an object type record
        description: Modify the name or description of an existing object type.
        parameters:
          - in: path
            name: obj_id
            type: string
            required: true
            description: UUID of the object type to update
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                name:
                  type: string
                  example: "Building"
                description:
                  type: string
                  example: "A man-made structure with walls and a roof"
        responses:
          200:
            description: Object type updated successfully
            schema:
              $ref: '#/definitions/ObjectType'
          403:
            description: Validation error
          404:
            description: Object type not found
        """
        data = request.get_json()
        try:
            data = obj_schema.load(data)
        except ValidationError as e:
            responseobject = {
                "status": "fail",
                "message": e.messages
            }
            return make_response(jsonify(responseobject), 403)
        obj = database.update(ObjectType, obj_id, **data)
        return obj_schema.dump(obj), 200
