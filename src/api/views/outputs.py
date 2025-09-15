#!/usr/bin/python3
"""Outputs views (predictions).

Routes:
 - GET  /api/v1/outputs
 - POST /api/v1/outputs
 - GET  /api/v1/outputs/<output_id>
 - PUT  /api/v1/outputs/<output_id>
 - DELETE /api/v1/outputs/<output_id>

Supports filtering by ai_model_id, label_id and input_id and uses pagination.
"""
from flask_restful import Resource
from flask import request, jsonify
from marshmallow import ValidationError, EXCLUDE
from sqlalchemy.orm import joinedload

from src import storage
from src.storage.outputs import Output
from src.api.serializers.outputs import OutputSchema

output_schema = OutputSchema(unknown=EXCLUDE)
outputs_schema = OutputSchema(many=True, unknown=EXCLUDE)


class OutputList(Resource):
    """List and create Outputs."""

    def get(self):
        """List outputs with optional filters and pagination."""
        sess = storage.database.session
        q = sess.query(Output).order_by(Output.created_at.desc())

        ai_model_id = request.args.get("ai_model_id")
        label_id = request.args.get("label_id")
        input_id = request.args.get("input_id")

        if ai_model_id:
            q = q.filter(Output.ai_model_id == ai_model_id)
        if label_id:
            q = q.filter(Output.label_id == label_id)
        if input_id:
            q = q.filter(Output.input_id == input_id)

        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        total = q.count()
        items = q.options(joinedload(Output.label), joinedload(Output.ai_model)).offset((page - 1) * per_page).limit(per_page).all()
        return {"page": page, "per_page": per_page, "total": total, "items": outputs_schema.dump(items)}, 200

    def post(self):
        """Create an output (prediction)."""
        sess = storage.database.session
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return {"message": "Invalid JSON payload"}, 400

        try:
            data = output_schema.load(payload, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        obj = Output(**data)
        sess.add(obj)
        sess.commit()
        return output_schema.dump(obj), 201


class OutputSingle(Resource):
    """Get, update or delete a single Output."""

    def get(self, output_id):
        """Retrieve an output by id."""
        sess = storage.database.session
        obj = sess.query(Output).get(output_id)
        if not obj:
            return {"message": "Output not found"}, 404
        return output_schema.dump(obj), 200

    def put(self, output_id):
        """Update an output."""
        sess = storage.database.session
        obj = sess.query(Output).get(output_id)
        if not obj:
            return {"message": "Output not found"}, 404

        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = output_schema.load(payload, partial=True, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        for k, v in data.items():
            setattr(obj, k, v)
        sess.add(obj)
        sess.commit()
        return output_schema.dump(obj), 200

    def delete(self, output_id):
        """Delete an output."""
        sess = storage.database.session
        obj = sess.query(Output).get(output_id)
        if not obj:
            return {"message": "Output not found"}, 404
        sess.delete(obj)
        sess.commit()
        return "", 204
