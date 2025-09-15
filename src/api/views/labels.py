#!/usr/bin/python3
"""Labels (object categories) views.

Routes:
 - GET  /api/v1/labels
 - POST /api/v1/labels
 - GET  /api/v1/labels/<label_id>
 - PUT  /api/v1/labels/<label_id>
 - DELETE /api/v1/labels/<label_id>

Label creation is validated by LabelSchema which enforces the safety blacklist
(e.g., blocks "military" or generic "vehicles" term by default).
"""
from flask_restful import Resource
from flask import request
from marshmallow import ValidationError, EXCLUDE

from src import storage
from src.storage.labels import Label
from src.api.serializers.labels import LabelSchema

label_schema = LabelSchema(unknown=EXCLUDE)
labels_schema = LabelSchema(many=True, unknown=EXCLUDE)


class LabelList(Resource):
    """List and create labels."""

    def get(self):
        """Return paginated list of labels."""
        sess = storage.database.session
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        q = sess.query(Label).order_by(Label.name.asc())
        total = q.count()
        items = q.offset((page - 1) * per_page).limit(per_page).all()
        return {"page": page, "per_page": per_page, "total": total, "items": labels_schema.dump(items)}, 200

    def post(self):
        """Create a label; LabelSchema enforces safety checks."""
        sess = storage.database.session
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return {"message": "Invalid JSON payload"}, 400

        # pass session so uniqueness checks are possible
        try:
            data = label_schema.load(payload, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        obj = Label(**data)
        sess.add(obj)
        sess.commit()
        return label_schema.dump(obj), 201


class LabelSingle(Resource):
    """Get/update/delete a single Label."""

    def get(self, label_id):
        """Retrieve a label by id."""
        sess = storage.database.session
        obj = sess.query(Label).get(label_id)
        if not obj:
            return {"message": "Label not found"}, 404
        return label_schema.dump(obj), 200

    def put(self, label_id):
        """Update label metadata."""
        sess = storage.database.session
        obj = sess.query(Label).get(label_id)
        if not obj:
            return {"message": "Label not found"}, 404

        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = label_schema.load(payload, partial=True, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        for k, v in data.items():
            setattr(obj, k, v)
        sess.add(obj)
        sess.commit()
        return label_schema.dump(obj), 200

    def delete(self, label_id):
        """Delete a label (will cascade to model_labels & outputs if configured)."""
        sess = storage.database.session
        obj = sess.query(Label).get(label_id)
        if not obj:
            return {"message": "Label not found"}, 404
        sess.delete(obj)
        sess.commit()
        return "", 204
