#!/usr/bin/python3
"""Output schema module.

Schema for Output objects (model predictions). Validates counts, confidence bounds,
and that referenced IDs (input_id, label_id, ai_model_id) are present and well-formed.
Optionally you can pass a SQLAlchemy session in context['session'] to perform
existence checks for the referenced rows.
"""
from marshmallow import Schema, fields, validates, validates_schema, ValidationError


class OutputSchema(Schema):
    """Schema for Output rows produced by models.

    Fields:
        id, created_at, updated_at: dump-only
        input_id, label_id, ai_model_id: required FK references (strings)
        predicted_count: integer >= 0 (defaults to 1)
        corrected_count: integer >= 0 (defaults to 0)
        confidence: float in [0.0, 1.0]
        bbox: optional JSON string describing bounding box
        is_human_verified: optional boolean
    Context options:
        - 'session': a SQLAlchemy session for optional existence checks
    """
    id = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    input_id = fields.Str(required=True)
    label_id = fields.Str(required=True)
    ai_model_id = fields.Str(required=True)

    predicted_count = fields.Integer(required=False)
    corrected_count = fields.Integer(required=False)
    confidence = fields.Float(required=False)
    bbox = fields.Str(required=False, allow_none=True)
    is_human_verified = fields.Boolean(required=False)

    @validates("predicted_count")
    def validate_predicted_count(self, value):
        if value is None:
            return
        if value < 0:
            raise ValidationError("predicted_count must be >= 0")

    @validates("corrected_count")
    def validate_corrected_count(self, value):
        if value is None:
            return
        if value < 0:
            raise ValidationError("corrected_count must be >= 0")

    @validates("confidence")
    def validate_confidence(self, value):
        if value is None:
            return
        if not (0.0 <= value <= 1.0):
            raise ValidationError("confidence must be between 0.0 and 1.0")

    @validates_schema
    def validate_references(self, data, **kwargs):
        """Optional: check that referenced FK rows exist in DB when a session is provided.

        Provide a SQLAlchemy session in schema.context['session'] to enable these checks.
        The checks avoid unnecessary DB access by only running when a `session` is provided.
        """
        sess = self.context.get("session")
        if sess is None:
            return  # nothing to check

        # Import models lazily to avoid circular imports during module load
        from src.storage.inputs import Input
        from src.storage.labels import Label
        from src.storage.ai_models import AIModel

        if not sess.query(Input).get(data.get("input_id")):
            raise ValidationError({"input_id": ["input_id does not reference an existing Input"]})
        if not sess.query(Label).get(data.get("label_id")):
            raise ValidationError({"label_id": ["label_id does not reference an existing Label"]})
        if not sess.query(AIModel).get(data.get("ai_model_id")):
            raise ValidationError({"ai_model_id": ["ai_model_id does not reference an existing AIModel"]})
