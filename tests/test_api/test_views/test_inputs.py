#!/usr/bin/python3
"""
Input Views module
"""
import os
from flask_restful import Resource
from flask import request, jsonify, make_response
from marshmallow import ValidationError, EXCLUDE
from src.storage import database, Input, ObjectType
from src.api.serializers.inputs import InputSchema
from src.api.utils.error_handlers import (
    create_error_response, handle_database_error, NotFoundAPIError
)

input_schema = InputSchema(unknown=EXCLUDE)
inputs_schema = InputSchema(many=True)


class InputList(Resource):
    """Handles multiple Inputs"""

    def get(self):
        """Retrieve all input records"""
        try:
            if database is None or not hasattr(database, 'all'):
                return [], 200

            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 20))
            per_page = max(1, min(per_page, 100))

            inputs = database.all(Input) or []
            total = len(inputs)

            start = (page - 1) * per_page
            end = start + per_page
            paged = inputs[start:end]

            enhanced = []
            for inp in paged:
                obj_type = database.get(ObjectType, id=inp.object_type_id)
                enhanced.append({
                    "id": inp.id,
                    "filename": inp.filename,
                    "image_path": inp.image_path,
                    "object_type": obj_type.name if obj_type else "Unknown",
                    "created_at": inp.created_at.isoformat() if getattr(inp, "created_at", None) else None,
                    "updated_at": inp.updated_at.isoformat() if getattr(inp, "updated_at", None) else None,
                })

            resp = make_response(jsonify(enhanced), 200)
            resp.headers["X-Total-Count"] = str(total)
            resp.headers["X-Page"] = str(page)
            resp.headers["X-Per-Page"] = str(per_page)
            return resp
        except Exception as e:
            return make_response(jsonify({
                "error": f"Failed to fetch results: {str(e)}",
                "results": []
            }), 200)

    def post(self):
        """Upload and create a new input"""
        try:
            if "file" not in request.files:
                return make_response(jsonify({"error": "No file provided"}), 400)

            file = request.files["file"]
            if file.filename == "":
                return make_response(jsonify({"error": "Empty filename"}), 400)

            object_type_id = request.form.get("object_type_id")
            if not object_type_id:
                return make_response(jsonify({"error": "object_type_id is required"}), 400)

            save_path = os.path.join("media", file.filename)
            os.makedirs("media", exist_ok=True)
            file.save(save_path)

            new_input = Input(
                filename=file.filename,
                image_path=save_path,
                object_type_id=object_type_id
            )
            new_input.save()

            response_data = {
                "id": str(new_input.id),
                "filename": new_input.filename,
                "image_path": new_input.image_path,
                "object_type_id": new_input.object_type_id,
                "created_at": new_input.created_at.isoformat() if getattr(new_input, "created_at", None) else None,
                "updated_at": new_input.updated_at.isoformat() if getattr(new_input, "updated_at", None) else None,
            }
            return make_response(jsonify(response_data), 201)

        except ValidationError as e:
            return make_response(jsonify({
                "status": "fail",
                "message": e.messages
            }), 403)
        except Exception as e:
            return make_response(jsonify({
                "error": f"Failed to process image: {str(e)}"
            }), 500)


class InputSingle(Resource):
    """Handles single Input operations"""

    def get(self, input_id):
        """Retrieve a single input by ID"""
        try:
            inp = database.get(Input, id=input_id)
            if not inp:
                return create_error_response(
                    NotFoundAPIError(
                        f"Input with ID {input_id} not found",
                        "The requested input record does not exist"
                    )
                )

            obj_type = database.get(ObjectType, id=inp.object_type_id)
            data = input_schema.dump(inp)
            data["object_type"] = obj_type.name if obj_type else "Unknown"
            data["created_at"] = inp.created_at.isoformat() if getattr(inp, "created_at", None) else None
            data["updated_at"] = inp.updated_at.isoformat() if getattr(inp, "updated_at", None) else None

            return make_response(jsonify(data), 200)
        except Exception as e:
            return handle_database_error(e)

    def delete(self, input_id):
        """Delete an input by ID"""
        inp = database.get(Input, id=input_id)
        database.delete(inp)
        return make_response(jsonify({"message": "resource successfully deleted"}), 200)
