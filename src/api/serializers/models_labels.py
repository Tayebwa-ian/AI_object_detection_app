#!/usr/bin/python3
"""ModelLabel schema module.

Schema for aggregated metrics per (AI model, Label) pair: accuracy, precision, recall, f1_score.
All metric values are expected to be in the range [0.0, 1.0].

Optionally accepts a SQLAlchemy session in schema.context['session'] to validate
that the referenced ai_model_id and label_id exist.
"""
from marshmallow import Schema, fields, validates, validates_schema, ValidationError


class ModelLabelSchema(Schema):
    """Schema for ModelLabel rows."""
    id = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    ai_model_id = fields.Str(required=True)
    label_id = fields.Str(required=True)

    accuracy = fields.Float(required=False)
    precision = fields.Float(required=False)
    recall = fields.Float(required=False)
    f1_score = fields.Float(required=False)

    @validates("accuracy")
    @validates("precision")
    @validates("recall")
    @validates("f1_score")
    def validate_metric_range(self, value):
        if value is None:
            return
        if not (0.0 <= value <= 1.0):
            raise ValidationError("metric values must be between 0.0 and 1.0")

    @validates_schema
    def validate_references(self, data, **kwargs):
        """Optional reference checks (requires context['session'])."""
        sess = self.context.get("session")
        if not sess:
            return
        from src.storage.ai_models import AIModel
        from src.storage.labels import Label

        if not sess.query(AIModel).get(data.get("ai_model_id")):
            raise ValidationError({"ai_model_id": ["ai_model_id does not reference an existing AIModel"]})
        if not sess.query(Label).get(data.get("label_id")):
            raise ValidationError({"label_id": ["label_id does not reference an existing Label"]})
