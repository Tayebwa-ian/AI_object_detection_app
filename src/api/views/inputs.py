#!/usr/bin/python3
"""Inputs views (Flask-RESTful resources).

Provides endpoints to create, list, update and delete Inputs.
Inputs represent an image + prompt that was submitted for inference.

Routes:
 - GET  /api/v1/inputs
 - POST /api/v1/inputs
 - GET  /api/v1/inputs/<input_id>
 - PUT  /api/v1/inputs/<input_id>
 - DELETE /api/v1/inputs/<input_id>
"""
from flask_restful import Resource
from flask import request, jsonify
from marshmallow import ValidationError, EXCLUDE
from sqlalchemy.orm import joinedload

from src import storage
from src.storage.inputs import Input
from src.api.serializers.inputs import InputSchema

input_schema = InputSchema(unknown=EXCLUDE)
inputs_schema = InputSchema(many=True, unknown=EXCLUDE)


class InputList(Resource):
    """List and create Inputs."""

    def get(self):
        """Return paginated list of inputs."""
        sess = storage.database.session
        try:
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 25))
        except ValueError:
            return {"message": "page and per_page must be integers"}, 400

        q = sess.query(Input).order_by(Input.created_at.desc())
        # optional filter
        if request.args.get("is_test") is not None:
            val = request.args.get("is_test").lower()
            if val in ("1", "true", "yes"):
                q = q.filter(Input.is_test.is_(True))
            elif val in ("0", "false", "no"):
                q = q.filter(Input.is_test.is_(False))

        total = q.count()
        items = q.offset((page - 1) * per_page).limit(per_page).all()
        return {"page": page, "per_page": per_page, "total": total, "items": inputs_schema.dump(items)}, 200

    def post(self):
        """Create a new Input. Validates payload using InputSchema."""
        sess = storage.database.session
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return {"message": "Invalid JSON payload"}, 400

        try:
            data = input_schema.load(payload, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        obj = Input(**data)
        sess.add(obj)
        sess.commit()
        return input_schema.dump(obj), 201


class InputSingle(Resource):
    """Retrieve, update, or delete a single Input."""

    def get(self, input_id):
        """Retrieve a single Input by id."""
        sess = storage.database.session
        obj = sess.query(Input).options(joinedload(Input.outputs)).get(input_id)
        if not obj:
            return {"message": "Input not found"}, 404
        return input_schema.dump(obj), 200

    def put(self, input_id):
        """Update an Input partially or fully."""
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "Input not found"}, 404

        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = input_schema.load(payload, partial=True, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        for k, v in data.items():
            setattr(obj, k, v)
        sess.add(obj)
        sess.commit()
        return input_schema.dump(obj), 200

    def delete(self, input_id):
        """Delete an Input."""
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "Input not found"}, 404
        sess.delete(obj)
        sess.commit()
        return "", 204
