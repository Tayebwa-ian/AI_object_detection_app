#!/usr/bin/python3
"""Input schema module"""
from marshmallow import Schema, fields


class InputSchema(Schema):
    """Input Schema 
            Responsible for data serialization, deserialization and validation
    """
    id = fields.Str(dump_only=True)
    created_at = fields.Str(dump_only=True)
    updated_at = fields.Str(dump_only=True)
    description = fields.Str(required=True)
    image_url = fields.Str(required=True)
