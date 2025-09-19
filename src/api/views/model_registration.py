#!/usr/bin/python3
"""Model registration view for CRUD operations on AIModel table."""

from flask_restful import Resource
from flask import request, jsonify
from sqlalchemy.exc import SQLAlchemyError

from src import storage
from src.storage.ai_models import AIModel


class AIModelList(Resource):
    """CRUD operations for multiple AI models."""

    def get(self):
        """
        Retrieve all AI models.
        ---
        tags:
          - AI Models
        responses:
          200:
            description: List of AI models
        """
        sess = storage.database.session
        models = sess.query(AIModel).all()
        return jsonify([{"id": m.id, "name": m.name, "description": m.description} for m in models])

    def post(self):
        """
        Insert a new AI model.
        ---
        tags:
          - AI Models
        parameters:
          - in: body
            name: model
            schema:
              type: object
              required:
                - name
              properties:
                name:
                  type: string
                description:
                  type: string
        responses:
          201:
            description: Created model
          400:
            description: Invalid input
        """
        data = request.get_json()
        if not data or "name" not in data:
            return {"error": "Missing required field 'name'"}, 400
        sess = storage.database.session
        try:
            model = AIModel(name=data["name"], description=data.get("description"))
            sess.add(model)
            sess.commit()
            return {"id": model.id, "name": model.name, "description": model.description}, 201
        except SQLAlchemyError as e:
            sess.rollback()
            return {"error": str(e)}, 400


class AIModelSingle(Resource):
    """CRUD operations for a single AI model."""

    def get(self, model_id):
        """
        Retrieve a model by ID.
        ---
        tags:
          - AI Models
        parameters:
          - in: path
            name: model_id
            type: string
            required: true
        responses:
          200:
            description: Model data
          404:
            description: Model not found
        """
        sess = storage.database.session
        model = sess.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return {"error": "Model not found"}, 404
        return {"id": model.id, "name": model.name, "description": model.description}

    def put(self, model_id):
        """
        Update a model by ID.
        ---
        tags:
          - AI Models
        parameters:
          - in: path
            name: model_id
            type: string
            required: true
          - in: body
            name: model
            schema:
              type: object
              properties:
                name:
                  type: string
                description:
                  type: string
        responses:
          200:
            description: Updated model
          404:
            description: Model not found
        """
        data = request.get_json()
        sess = storage.database.session
        model = sess.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return {"error": "Model not found"}, 404
        if "name" in data:
            model.name = data["name"]
        if "description" in data:
            model.description = data["description"]
        try:
            sess.commit()
            return {"id": model.id, "name": model.name, "description": model.description}
        except SQLAlchemyError as e:
            sess.rollback()
            return {"error": str(e)}, 400

    def delete(self, model_id):
        """
        Delete a model by ID.
        ---
        tags:
          - AI Models
        parameters:
          - in: path
            name: model_id
            type: string
            required: true
        responses:
          200:
            description: Model deleted
          404:
            description: Model not found
        """
        sess = storage.database.session
        model = sess.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return {"error": "Model not found"}, 404
        try:
            sess.delete(model)
            sess.commit()
            return {"message": "Model deleted"}
        except SQLAlchemyError as e:
            sess.rollback()
            return {"error": str(e)}, 400
