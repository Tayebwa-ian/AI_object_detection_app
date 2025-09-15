#!/usr/bin/python3
"""Label schema module.

Schema for Label entries (categories). Implements domain safety checks using
a configurable blacklist. By default, military/weapon-related labels and the
generic 'vehicles' label are rejected.
"""
from marshmallow import Schema, fields, validates, ValidationError


DEFAULT_BANNED_KEYWORDS = [
    "military", "weapon", "weapons", "tank", "missile", "firearm", "firearms",
    "bomb", "grenade", "warship", "naval", "fighter", "aircraft"
]


class LabelSchema(Schema):
    """Schema for Label objects.

    Fields:
        id, created_at, updated_at: dump-only
        name: required label name (2-128 chars). Will be checked against safety blacklist.
        description: optional short description (up to 1024 chars).
    Context options (optional):
        - 'banned_label_keywords': list of strings to treat as substrings to block
        - 'allow_generic_labels': if True the generic label 'vehicles' will be allowed
        - 'session': optional SQLAlchemy session for uniqueness checks (name)
    """
    id = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    name = fields.Str(required=True)
    description = fields.Str(required=False, allow_none=True)

    @validates("name")
    def validate_name_and_safety(self, value):
        """Validate name length and check safety blacklists.

        Behavior:
          - Short/long names rejected.
          - If `banned_label_keywords` provided in schema.context, use that list.
            Otherwise use DEFAULT_BANNED_KEYWORDS.
          - By default the generic label 'vehicles' is rejected (user request). If
            you explicitly set context['allow_generic_labels'] = True, it will be allowed.
          - Optionally checks for uniqueness if a SQLAlchemy session is provided as context['session'].
        """
        v = (value or "").strip()
        if not (1 < len(v) <= 128):
            raise ValidationError("name must be between 2 and 128 characters")

        lower = v.lower()

        # Custom banned list
        banned = self.context.get("banned_label_keywords", DEFAULT_BANNED_KEYWORDS)

        # block if any banned keyword appears as a substring in the name
        for kw in banned:
            if kw and kw.lower() in lower:
                raise ValidationError(f"label name contains disallowed keyword: '{kw}'")

        # The user explicitly asked to catch "vehicles" as an example.
        # By default reject the generic 'vehicles' term unless context allows it.
        allow_generic = bool(self.context.get("allow_generic_labels", False))
        if "vehicles" in lower and not allow_generic:
            raise ValidationError("generic label 'vehicles' is not allowed by policy; use a more specific label")

        # Optional uniqueness check (requires a SQLAlchemy session in context)
        sess = self.context.get("session")
        if sess is not None:
            from src.storage.labels import Label
            existing = sess.query(Label).filter(Label.name == v).one_or_none()
            if existing:
                raise ValidationError("a label with this name already exists")
