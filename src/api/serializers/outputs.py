#!/usr/bin/python3
"""Output schema module"""
from marshmallow import Schema, fields


class OutputSchema(Schema):
    """Output Schema
            Responsible for data serialization, deserialization and validation
    """
    id = fields.Str(dump_only=True)
    created_at = fields.Str(dump_only=True)
    updated_at = fields.Str(dump_only=True)
    predicted_count = fields.Integer(required=True)
    corrected_count = fields.Integer()
    pred_confidence = fields.Float(required=True)
    object_type_id = fields.Str(required=True)
    input_id = fields.Str(required=True)
