#!/usr/bin/python3
"""InferencePeriod schema module.

Schema for latency (inference) records. Value should be a positive float (seconds or milliseconds
depending on your convention). Optional FK existence checks can be enabled by passing a SQLAlchemy
session in schema.context['session'].
"""
from marshmallow import Schema, fields, validates, validates_schema, ValidationError


class InferencePeriodSchema(Schema):
    """Schema for InferencePeriod rows."""
    id = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    ai_model_id = fields.Str(required=True)
    input_id = fields.Str(required=True)
    value = fields.Float(required=True)

    @validates("value")
    def validate_value(self, v):
        if v is None:
            raise ValidationError("value is required")
        if v < 0.0:
            raise ValidationError("value must be non-negative (latency in seconds or ms)")

    @validates_schema
    def validate_references(self, data, **kwargs):
        """Optional FK existence validation using SQLAlchemy session in context['session']."""
        sess = self.context.get("session")
        if not sess:
            return
        from src.storage.ai_models import AIModel
        from src.storage.inputs import Input

        if not sess.query(AIModel).get(data.get("ai_model_id")):
            raise ValidationError({"ai_model_id": ["ai_model_id does not reference an existing AIModel"]})
        if not sess.query(Input).get(data.get("input_id")):
            raise ValidationError({"input_id": ["input_id does not reference an existing Input"]})
