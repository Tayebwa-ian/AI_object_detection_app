#!/usr/bin/python3
"""
Labels (object categories) views.

Routes:
 - GET    /api/v1/labels
 - POST   /api/v1/labels
 - GET    /api/v1/labels/<label_id>
 - PUT    /api/v1/labels/<label_id>
 - DELETE /api/v1/labels/<label_id>

Notes:
------
- Label creation is validated via LabelSchema, which enforces safety blacklist
  (e.g., blocks "military" or generic "vehicles" term by default).
- Deletion may cascade to ModelLabel and Output if database constraints are set.
- Supports pagination on listing.
"""

from flask_restful import Resource
from flask import request
from marshmallow import ValidationError, EXCLUDE

from src import storage
from src.storage.labels import Label
from src.api.serializers.labels import LabelSchema

# Schemas
label_schema = LabelSchema(unknown=EXCLUDE)
labels_schema = LabelSchema(many=True, unknown=EXCLUDE)


# -----------------------------
# Helper functions
# -----------------------------
def _get_label_or_404(sess, label_id):
    """
    Retrieve a Label object or return a 404 response.
    
    Parameters
    ----------
    sess : SQLAlchemy session
    label_id : str
        The primary key of the label.
    
    Returns
    -------
    Label object or tuple(response, 404)
    """
    obj = sess.query(Label).get(label_id)
    if not obj:
        return None, {"message": "Label not found"}, 404
    return obj, None, None


# -----------------------------
# Resources
# -----------------------------
class LabelList(Resource):
    """List and create labels."""

    def get(self):
        """
        Retrieve paginated list of labels.
        ---
        tags:
          - Labels
        parameters:
          - in: query
            name: page
            schema:
              type: integer
              default: 1
            description: Page number
          - in: query
            name: per_page
            schema:
              type: integer
              default: 50
            description: Items per page
        responses:
          200:
            description: List of labels with pagination
        """
        sess = storage.database.session
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))

        q = sess.query(Label).order_by(Label.name.asc())
        total = q.count()
        items = q.offset((page - 1) * per_page).limit(per_page).all()

        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "items": labels_schema.dump(items),
        }, 200

    def post(self):
        """
        Create a new label.
        ---
        tags:
          - Labels
        requestBody:
          required: true
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Label'
        responses:
          201:
            description: Label created successfully
          400:
            description: Invalid JSON
          422:
            description: Validation error (blacklist, uniqueness, etc.)
        """
        sess = storage.database.session
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return {"message": "Invalid JSON payload"}, 400

        try:
            # Context passes session for uniqueness and safety checks
            data = label_schema.load(payload, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        obj = Label(**data)
        sess.add(obj)
        sess.commit()
        return label_schema.dump(obj), 201


class LabelSingle(Resource):
    """Retrieve, update, or delete a single label."""

    def get(self, label_id):
        """
        Retrieve a label by ID.
        ---
        tags:
          - Labels
        parameters:
          - in: path
            name: label_id
            required: true
            schema:
              type: string
        responses:
          200:
            description: Label object
          404:
            description: Label not found
        """
        sess = storage.database.session
        obj, resp, status = _get_label_or_404(sess, label_id)
        if resp:
            return resp, status
        return label_schema.dump(obj), 200

    def put(self, label_id):
        """
        Update a label's metadata (name, description, etc.).
        ---
        tags:
          - Labels
        parameters:
          - in: path
            name: label_id
            required: true
            schema:
              type: string
        requestBody:
          required: true
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Label'
        responses:
          200:
            description: Updated label
          404:
            description: Label not found
          422:
            description: Validation error
        """
        sess = storage.database.session
        obj, resp, status = _get_label_or_404(sess, label_id)
        if resp:
            return resp, status

        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = label_schema.load(payload, partial=True, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        # Apply allowed updates
        for k, v in data.items():
            setattr(obj, k, v)

        sess.add(obj)
        sess.commit()
        return label_schema.dump(obj), 200

    def delete(self, label_id):
        """
        Delete a label by ID.
        ---
        tags:
          - Labels
        parameters:
          - in: path
            name: label_id
            required: true
            schema:
              type: string
        responses:
          204:
            description: Label deleted successfully
          404:
            description: Label not found
        """
        sess = storage.database.session
        obj, resp, status = _get_label_or_404(sess, label_id)
        if resp:
            return resp, status

        # Deletion cascades to related model_labels or outputs if DB configured
        sess.delete(obj)
        sess.commit()
        return "", 204
