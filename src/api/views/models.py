#!/usr/bin/python3
"""AIModel views: CRUD + training/testing endpoints.

Endpoints:
 - GET  /api/v1/models
 - POST /api/v1/models
 - GET  /api/v1/models/<id>
 - PUT  /api/v1/models/<id>
 - DELETE /api/v1/models/<id>
 - POST /api/v1/models/<id>/train   (simulate a training run)
 - POST /api/v1/models/<id>/test    (submit or run a test; persists ModelLabel & InferencePeriod)
"""
from flask_restful import Resource
from flask import request, jsonify
from marshmallow import ValidationError, EXCLUDE
from datetime import datetime
import json

from src import storage
from src.storage.ai_models import AIModel
from src.storage.evaluation_runs import EvaluationRun
from src.storage.models_labels import ModelLabel
from src.storage.inference_periods import InferencePeriod
from src.api.serializers.ai_modles import AIModelSchema
from src.api.serializers.models_labels import ModelLabelSchema
from src.api.serializers.inference_periods import InferencePeriodSchema

# Schemas
ai_model_schema = AIModelSchema(unknown=EXCLUDE)
ai_models_schema = AIModelSchema(many=True, unknown=EXCLUDE)
model_label_schema = ModelLabelSchema(unknown=EXCLUDE)
inference_period_schema = InferencePeriodSchema(unknown=EXCLUDE)


class AIModelList(Resource):
    """List/create AI models."""

    def get(self):
        """Return list of models."""
        sess = storage.database.session
        models = sess.query(AIModel).order_by(AIModel.name.asc()).all()
        return ai_models_schema.dump(models), 200

    def post(self):
        """Create a new AI model."""
        sess = storage.database.session
        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = ai_model_schema.load(payload, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        model = AIModel(**data)
        sess.add(model)
        sess.commit()
        return ai_model_schema.dump(model), 201


class AIModelSingle(Resource):
    """Get/update/delete a single AI model."""

    def get(self, model_id):
        """Retrieve a model by id."""
        sess = storage.database.session
        obj = sess.query(AIModel).get(model_id)
        if not obj:
            return {"message": "AIModel not found"}, 404
        return ai_model_schema.dump(obj), 200

    def put(self, model_id):
        """Update model metadata."""
        sess = storage.database.session
        obj = sess.query(AIModel).get(model_id)
        if not obj:
            return {"message": "AIModel not found"}, 404
        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = ai_model_schema.load(payload, partial=True, context={"session": sess})
        except ValidationError as exc:
            return {"message": "Validation error", "errors": exc.messages}, 422

        for k, v in data.items():
            setattr(obj, k, v)
        sess.add(obj)
        sess.commit()
        return ai_model_schema.dump(obj), 200

    def delete(self, model_id):
        """Delete an AI model and cascade related rows."""
        sess = storage.database.session
        obj = sess.query(AIModel).get(model_id)
        if not obj:
            return {"message": "AIModel not found"}, 404
        sess.delete(obj)
        sess.commit()
        return "", 204


class TrainModel(Resource):
    """Trigger a training run (development/demo synchronous simulation).

    For production you should enqueue a job into a worker (Celery/RQ) and return the created run id.
    This endpoint simulates training and returns a created EvaluationRun record.
    """

    def post(self, model_id):
        """
        Start a training run for the model.
        Request JSON (example):
        {
            "dataset": "my_dataset_v1",
            "epochs": 10,
            "params": {"lr": 0.001, "batch_size": 16}
        }
        """
        sess = storage.database.session
        model = sess.query(AIModel).get(model_id)
        if not model:
            return {"message": "AIModel not found"}, 404

        payload = request.get_json(force=True, silent=True) or {}
        dataset = payload.get("dataset")
        epochs = int(payload.get("epochs", 0))
        params = payload.get("params", {})

        # Create a run record (simulate start/finish immediately for dev)
        metadata = json.dumps({"dataset": dataset, "epochs": epochs, "params": params})
        run = EvaluationRun(ai_model_id=model_id, run_type="train", meta_data=metadata)
        run.started_at = datetime.now()
        run.finished_at = datetime.now()
        sess.add(run)
        sess.commit()

        return {"message": "training run recorded (simulated)", "run_id": run.id, "metadata": json.loads(run.metadata)}, 201


class TestModel(Resource):
    """Run or submit test results for a given model and persist them.

    This endpoint accepts a test payload containing per-label metrics and per-input latencies.
    All ModelLabel and InferencePeriod rows created here will be linked to a new EvaluationRun.
    """

    def post(self, model_id):
        """
        Submit test results.

        Example payload:
        {
          "dataset": "testset-v1",
          "results": [
            {"label_id": "<label-uuid>", "accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1_score": 0.95},
            ...
          ],
          "latencies": [
            {"input_id": "<input-uuid>", "value": 0.123},
            ...
          ],
          "metadata": {"notes": "batch test"}
        }

        The endpoint validates that referenced label_ids and input_ids exist (requires DB session).
        """
        sess = storage.database.session
        model = sess.query(AIModel).get(model_id)
        if not model:
            return {"message": "AIModel not found"}, 404

        payload = request.get_json(force=True, silent=True) or {}
        dataset = payload.get("dataset")
        results = payload.get("results", [])
        latencies = payload.get("latencies", [])
        metadata_extra = payload.get("metadata", {})

        # create evaluation run (test)
        run_metadata = json.dumps({"dataset": dataset, "metadata": metadata_extra})
        run = EvaluationRun(ai_model_id=model_id, run_type="test", meta_data=run_metadata)
        run.started_at = datetime.now()
        sess.add(run)
        sess.flush()  # get run.id without final commit

        # Validate and insert ModelLabel rows
        inserted_ml = []
        for r in results:
            # attach run_id & model_id
            r_payload = {
                "ai_model_id": model_id,
                "label_id": r.get("label_id"),
                "accuracy": r.get("accuracy", 0.0),
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
                "f1_score": r.get("f1_score", 0.0),
                "run_id": run.id
            }
            try:
                ml_data = model_label_schema.load(r_payload, context={"session": sess})
            except ValidationError as exc:
                sess.rollback()
                return {"message": "ModelLabel validation error", "errors": exc.messages, "failed_payload": r_payload}, 422

            ml = ModelLabel(**ml_data)
            sess.add(ml)
            inserted_ml.append(ml)

        # Validate and insert InferencePeriod rows (latencies)
        inserted_lat = []
        for l in latencies:
            l_payload = {
                "ai_model_id": model_id,
                "input_id": l.get("input_id"),
                "value": l.get("value", 0.0),
                "run_id": run.id
            }
            try:
                lat_data = inference_period_schema.load(l_payload, context={"session": sess})
            except ValidationError as exc:
                sess.rollback()
                return {"message": "InferencePeriod validation error", "errors": exc.messages, "failed_payload": l_payload}, 422

            ip = InferencePeriod(**lat_data)
            sess.add(ip)
            inserted_lat.append(ip)

        # Optionally you could persist raw outputs if payload contains them (not implemented here)
        run.finished_at = datetime.now()
        sess.commit()

        return {
            "message": "test results stored",
            "run_id": run.id,
            "model_labels_inserted": len(inserted_ml),
            "latencies_inserted": len(inserted_lat)
        }, 201
