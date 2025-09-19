#!/usr/bin/python3
"""
Outputs views (predictions).

Routes:
 - GET    /api/v1/outputs
 - GET    /api/v1/outputs/<output_id>
 - PUT    /api/v1/outputs/<output_id>
 - DELETE /api/v1/outputs/<output_id>

Notes:
------
- Outputs are generated automatically from the AI pipeline and stored in DB.
- Users cannot create outputs manually via API (`POST` removed).
- Users may update certain fields (e.g., corrected counts) if predictions are wrong.
- Supports filtering by `ai_model_id`, `label_id`, `input_id` and uses pagination.
"""

from flask_restful import Resource
from flask import request
from marshmallow import ValidationError, EXCLUDE
from sqlalchemy.orm import joinedload

from src import storage
from src.storage.outputs import Output
from src.api.serializers.outputs import OutputSchema

# Marshmallow schemas
output_schema = OutputSchema(unknown=EXCLUDE)
outputs_schema = OutputSchema(many=True, unknown=EXCLUDE)


# -----------------------------
# Helper functions
# -----------------------------
def _apply_corrections(obj: Output, data: dict) -> None:
    """
    Apply user-provided corrections to an Output object.

    Only specific fields are editable by users:
    - corrected_count: int (user-corrected object count for this label)

    Parameters
    ----------
    obj : Output
        The Output object to update.
    data : dict
        Validated payload containing correction fields.
    """
    if "corrected_count" in data:
        obj.corrected_count = data["corrected_count"]


# -----------------------------
# Resources
# -----------------------------
class OutputList(Resource):
    """List Outputs (predictions)."""

    def get(self):
        """
        List outputs with optional filters and pagination.
        ---
        tags:
          - Outputs
        parameters:
          - in: query
            name: ai_model_id
            schema:
              type: integer
            description: Filter by AI model id
          - in: query
            name: label_id
            schema:
              type: integer
            description: Filter by Label id
          - in: query
            name: input_id
            schema:
              type: integer
            description: Filter by Input id
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
            description: Paginated list of outputs
        """
        sess = storage.database.session
        q = sess.query(Output).order_by(Output.created_at.desc())

        # Filtering
        ai_model_id = request.args.get("ai_model_id")
        label_id = request.args.get("label_id")
        input_id = request.args.get("input_id")

        if ai_model_id:
            q = q.filter(Output.ai_model_id == ai_model_id)
        if label_id:
            q = q.filter(Output.label_id == label_id)
        if input_id:
            q = q.filter(Output.input_id == input_id)

        # Pagination
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        total = q.count()

        items = (
            q.options(joinedload(Output.label), joinedload(Output.ai_model))
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "items": outputs_schema.dump(items),
        }, 200


class OutputSingle(Resource):
    """Retrieve, update, or delete a single Output."""

    def get(self, output_id):
        """
        Retrieve an output by ID.
        ---
        tags:
          - Outputs
        parameters:
          - in: path
            name: output_id
            required: true
            schema:
              type: integer
        responses:
          200:
            description: Output object
          404:
            description: Output not found
        """
        sess = storage.database.session
        obj = sess.query(Output).get(output_id)
        if not obj:
            return {"message": "Output not found"}, 404
        return output_schema.dump(obj), 200

    def put(self, output_id):
        """
        Update an output (e.g., corrected count).
        ---
        tags:
          - Outputs
        parameters:
          - in: path
            name: output_id
            required: true
            schema:
              type: integer
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
                properties:
                  corrected_count:
                    type: integer
                    description: Corrected object count provided by the user
        responses:
          200:
            description: Updated Output object
          404:
            description: Output not found
          422:
            description: Validation error
        """
        sess = storage.database.session
        obj = sess.query(Output).get(output_id)
        if not obj:
            return {"message": "Output not found"}, 404

        # Parse and validate JSON payload
        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = output_schema.load(
                payload, partial=True, context={"session": sess}
            )
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        # Apply corrections (only allowed fields updated)
        _apply_corrections(obj, data)

        sess.add(obj)
        sess.commit()
        return output_schema.dump(obj), 200

    def delete(self, output_id):
        """
        Delete an output by ID.
        ---
        tags:
          - Outputs
        parameters:
          - in: path
            name: output_id
            required: true
            schema:
              type: integer
        responses:
          204:
            description: Output deleted successfully
          404:
            description: Output not found
        """
        sess = storage.database.session
        obj = sess.query(Output).get(output_id)
        if not obj:
            return {"message": "Output not found"}, 404

        sess.delete(obj)
        sess.commit()
        return "", 204
