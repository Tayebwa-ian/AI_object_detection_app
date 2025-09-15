#!/usr/bin/python3
"""Input schema module.

Schema for Input objects (what the user submits for inference).
Provides validation for prompt and image_path (accepts URLs or local filenames with common image extensions).
Also validates learning-approach flags (zero-shot / few-shot) are mutually exclusive.
"""
import re
from marshmallow import Schema, fields, validates, validates_schema, ValidationError


# regex for a simple path that ends with a common image extension
_IMAGE_PATH_RE = re.compile(r".+\.(jpg|jpeg|png|bmp|gif|tiff|webp)$", re.IGNORECASE)


class InputSchema(Schema):
    """Schema for Input objects.

    Fields:
        id, created_at, updated_at: dump-only timestamps and id.
        prompt: textual prompt given to the model (optional but recommended).
        image_path: URL or local filename/relative path to the image. Required.
        violation_count: non-negative int.
        is_zero_shot / is_few_shot / is_test: booleans (is_zero_shot and is_few_shot can't both be true).
    """
    id = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    prompt = fields.Str(required=False, allow_none=True)
    image_path = fields.Str(required=True,
                            metadata={"description": "Image URL or path (jpg/png/etc.)"})

    violation_count = fields.Integer(required=False)
    is_zero_shot = fields.Boolean(required=False)
    is_few_shot = fields.Boolean(required=False)
    is_test = fields.Boolean(required=False)

    @validates("image_path")
    def validate_image_path(self, value):
        """Validate that image_path is either a URL or a filename/path ending with an image extension.

        Accept either:
          - A fully-qualified URL (http[s]://...) OR
          - A local/relative path / filename that looks like an image (ends with .jpg/.png/etc.)
        """
        v = (value or "").strip()
        if not v:
            raise ValidationError("image_path cannot be empty")

        # Quick url check
        if v.startswith("http://") or v.startswith("https://"):
            # Accept it as URL. Optionally you could perform deeper URL validation.
            return

        # Otherwise enforce filename extension pattern
        if not _IMAGE_PATH_RE.match(v):
            raise ValidationError("image_path must be a URL or a path that ends with a common image extension "
                                  "(jpg, jpeg, png, bmp, gif, tiff, webp)")

    @validates("violation_count")
    def validate_violation_count(self, value):
        if value is None:
            return
        if value < 0:
            raise ValidationError("violation_count cannot be negative")

    @validates_schema
    def validate_shot_flags(self, data, **kwargs):
        """Make sure both is_zero_shot and is_few_shot are not True simultaneously."""
        if data.get("is_zero_shot") and data.get("is_few_shot"):
            raise ValidationError("is_zero_shot and is_few_shot cannot both be true")
