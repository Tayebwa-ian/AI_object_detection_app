# src/docs/swagger_template.py

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "AI Object Counter API",
        "description": "Comprehensive API for AI-powered object detection, counting, and few-shot learning. Supports image processing, batch operations, performance monitoring, and real-time corrections.",
        "version": "2.0.0",
        "contact": {
            "name": "AI Object Counter Team",
            "email": "support@aiobjectcounter.com"
        }
    },
    "basePath": "/api",
    "schemes": ["http", "https"],
    "consumes": ["application/json", "multipart/form-data"],
    "produces": ["application/json"],
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header"
        }
    },
    "definitions": {
        "Input": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "example": "123e4567-e89b-12d3-a456-426614174000"},
                "description": {"type": "string", "example": "A photo of a dog in a park"},
                "image_path": {"type": "string", "example": "/media/dog.jpg"},
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
                "object_type": {"type": "string", "example": "car"},
                "image_path": {"type": "string", "example": "/media/car.jpg"},
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
        },
        "CountResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "result_id": {"type": "string", "example": "789e1234-e89b-12d3-a456-426614174000"},
                "object_type": {"type": "string", "example": "car"},
                "predicted_count": {"type": "integer", "example": 3},
                "confidence": {"type": "number", "format": "float", "example": 0.95},
                "processing_time": {"type": "number", "format": "float", "example": 2.5},
                "image_path": {"type": "string", "example": "/media/car.jpg"},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"}
            }
        },
        "CorrectionRequest": {
            "type": "object",
            "properties": {
                "corrected_count": {"type": "integer", "example": 4, "description": "The corrected count provided by the user"}
            },
            "required": ["corrected_count"]
        },
        "CorrectionResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "result_id": {"type": "string", "example": "789e1234-e89b-12d3-a456-426614174000"},
                "predicted_count": {"type": "integer", "example": 3},
                "corrected_count": {"type": "integer", "example": 4},
                "updated_at": {"type": "string", "format": "date-time", "example": "2025-09-03T13:00:00"},
                "message": {"type": "string", "example": "Correction submitted successfully"}
            }
        },
        "BatchProcessingRequest": {
            "type": "object",
            "properties": {
                "images": {
                    "type": "array",
                    "items": {"type": "file"},
                    "description": "Multiple image files to process"
                },
                "object_type": {"type": "string", "example": "car", "description": "Type of object to count"},
                "description": {"type": "string", "example": "Batch processing of car images", "description": "Optional description for the batch"},
                "auto_detect": {"type": "boolean", "example": False, "description": "Enable auto-detection mode"}
            },
            "required": ["images", "object_type"]
        },
        "BatchProcessingResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "batch_id": {"type": "string", "example": "batch-123e4567-e89b-12d3-a456-426614174000"},
                "total_images": {"type": "integer", "example": 5},
                "processed_images": {"type": "integer", "example": 5},
                "failed_images": {"type": "integer", "example": 0},
                "results": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/CountResponse"}
                },
                "processing_time": {"type": "number", "format": "float", "example": 12.5},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"}
            }
        },
        "FewShotRegisterRequest": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "example": "elephant", "description": "Name of the new object type"},
                "support_images": {
                    "type": "array",
                    "items": {"type": "file"},
                    "description": "Support images for few-shot learning (minimum 3 recommended)"
                },
                "description": {"type": "string", "example": "Large mammal with trunk", "description": "Description of the object type"}
            },
            "required": ["object_name", "support_images"]
        },
        "FewShotRegisterResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "object_name": {"type": "string", "example": "elephant"},
                "support_images_count": {"type": "integer", "example": 5},
                "prototype_features": {"type": "array", "items": {"type": "number"}, "description": "Learned prototype features"},
                "confidence_threshold": {"type": "number", "format": "float", "example": 0.7},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"}
            }
        },
        "PerformanceMetrics": {
            "type": "object",
            "properties": {
                "total_requests": {"type": "integer", "example": 150},
                "successful_requests": {"type": "integer", "example": 142},
                "failed_requests": {"type": "integer", "example": 8},
                "success_rate": {"type": "number", "format": "float", "example": 0.947},
                "average_processing_time": {"type": "number", "format": "float", "example": 2.3},
                "uptime_seconds": {"type": "number", "format": "float", "example": 3600},
                "object_type_stats": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer", "example": 25},
                            "total_time": {"type": "number", "format": "float", "example": 57.5},
                            "successes": {"type": "integer", "example": 23},
                            "failures": {"type": "integer", "example": 2},
                            "average_time": {"type": "number", "format": "float", "example": 2.3}
                        }
                    }
                }
            }
        },
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "error": {"type": "string", "example": "Bad request"},
                "message": {"type": "string", "example": "Detailed error message"},
                "status": {"type": "string", "example": "fail"},
                "details": {"type": "object", "description": "Additional error details"}
            }
        },
        "HealthResponse": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "example": "healthy"},
                "message": {"type": "string", "example": "AI Object Counting API is running"},
                "pipeline_available": {"type": "boolean", "example": True},
                "database": {"type": "string", "example": "connected"},
                "timestamp": {"type": "string", "format": "date-time", "example": "2025-09-03T12:34:56"}
            }
        }
    },
    "tags": [
        {
            "name": "Inputs",
            "description": "Image upload and processing endpoints"
        },
        {
            "name": "Outputs", 
            "description": "Prediction results and corrections"
        },
        {
            "name": "Object Types",
            "description": "Object type management"
        },
        {
            "name": "Batch Processing",
            "description": "Batch image processing operations"
        },
        {
            "name": "Few-Shot Learning",
            "description": "Few-shot learning and custom object types"
        },
        {
            "name": "Performance Monitoring",
            "description": "System performance and metrics"
        },
        {
            "name": "System",
            "description": "System health and status endpoints"
        }
    ]
}
