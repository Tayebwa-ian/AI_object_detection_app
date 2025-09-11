# src/docs/swagger_template.py

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Object Detection API",
        "description": "API for managing inputs, outputs, and object types.",
        "version": "1.0.0"
    },
    "basePath": "/api",
    "schemes": ["http"],
    "definitions": {
        "Input": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "example": "123e4567-e89b-12d3-a456-426614174000"},
                "description": {"type": "string", "example": "A photo of a dog in a park"},
                "image_path": {"type": "string", "example": "/uploads/dog.jpg"},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"},
                "updated_at": {"type": "string", "format": "date-time", "example": "2025-09-03T13:00:00"}
            },
            "required": ["description", "image_path"]
        },
        "Output": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "example": "789e1234-e89b-12d3-a456-426614174000"},
                "predicted_count": {"type": "integer", "example": 5},
                "corrected_count": {"type": "integer", "example": 4},
                "pred_confidence": {"type": "number", "format": "float", "example": 0.92},
                "object_type_id": {"type": "string", "example": "object-type-uuid"},
                "input_id": {"type": "string", "example": "input-uuid"},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"},
                "updated_at": {"type": "string", "format": "date-time", "example": "2025-09-03T13:00:00"}
            },
            "required": ["predicted_count", "pred_confidence", "object_type_id", "input_id"]
        },
        "ObjectType": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "example": "456e7890-e89b-12d3-a456-426614174000"},
                "name": {"type": "string", "example": "Car"},
                "description": {"type": "string", "example": "A four-wheeled vehicle"},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"},
                "updated_at": {"type": "string", "format": "date-time", "example": "2025-09-03T13:00:00"}
            },
            "required": ["name", "description"]
        }
    }
}
