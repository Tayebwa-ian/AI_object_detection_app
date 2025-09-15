#!/usr/bin/python3
"""AIModel schema module.

Schema for the AIModel table. Validates name and description, and optionally
checks uniqueness using a SQLAlchemy session provided in schema.context['session'].
"""
from marshmallow import Schema, fields, validates, ValidationError


class AIModelSchema(Schema):
    """Schema for AIModel objects.

    Fields:
        id, created_at, updated_at: output-only fields provided by the model.
        name: required, 2-128 chars, optionally unique (requires session in context).
        description: optional, up to 1024 chars.
    """
    id = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    name = fields.Str(required=True,
                      metadata={"description": "Human readable unique model name"})
    description = fields.Str(required=False, allow_none=True)

    @validates("name")
    def validate_name(self, value):
        """Validate model name length and optionally check uniqueness.

        If a SQLAlchemy session is provided in context under 'session' then uniqueness
        against the `ai_models` table is checked. Avoid DB calls when not necessary.
        """
        v = (value or "").strip()
        if not (2 <= len(v) <= 128):
            raise ValidationError("name must be between 2 and 128 characters")

        # optional uniqueness check via SQLAlchemy session (context['session'])
        sess = self.context.get("session")
        if sess is not None:
            # perform lightweight uniqueness check. Import here to avoid circular imports
            from src.storage.ai_models import AIModel
            existing = sess.query(AIModel).filter(AIModel.name == v).one_or_none()
            if existing:
                raise ValidationError("an AI model with this name already exists")
