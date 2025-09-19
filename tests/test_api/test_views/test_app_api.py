from flask import request, jsonify
from flask.views import MethodView
from src import storage
import uuid

# -------------------------
# AI Model List Endpoint
# -------------------------
class AIModelList(MethodView):
    """Handles GET /models and POST /models"""

    def get(self):
        models = storage.all(cls=None)  # cls=None returns all AI models
        result = [
            {"id": m.id, "name": m.name, "description": getattr(m, "description", None)}
            for m in models
        ]
        return jsonify(result), 200

    def post(self):
        data = request.get_json(force=True) or {}
        if "name" not in data:
            return jsonify({"error": "Missing required field 'name'"}), 400

        model_id = str(uuid.uuid4())
        model_obj = type("AIModelMock", (), {})()
        model_obj.id = model_id
        model_obj.name = data["name"]
        model_obj.description = data.get("description")

        storage.new(model_obj)
        storage.save()
        return jsonify({"id": model_obj.id, "name": model_obj.name, "description": model_obj.description}), 201


# -------------------------
# Single AI Model Endpoint
# -------------------------
class AIModelSingle(MethodView):
    """Handles GET, PUT, DELETE /models/<model_id>"""

    def get(self, model_id):
        model = storage.get(cls=None, id=model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        return jsonify({"id": model.id, "name": model.name, "description": getattr(model, "description", None)}), 200

    def put(self, model_id):
        model = storage.get(cls=None, id=model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        data = request.get_json(force=True) or {}
        if "name" in data:
            model.name = data["name"]
        if "description" in data:
            model.description = data["description"]

        storage.save()
        return jsonify({"id": model.id, "name": model.name, "description": getattr(model, "description", None)}), 200

    def delete(self, model_id):
        model = storage.get(cls=None, id=model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        storage.delete(model)
        return jsonify({"message": "Model deleted"}), 200
