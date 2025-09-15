#!/usr/bin/python3
"""
AI Model Views - manage AIModel CRUD and run training/testing workflows.

Endpoints:
 - GET  /api/v1/models           list models
 - POST /api/v1/models           create an AI model
 - GET/PUT/DELETE /api/v1/models/<id>
 - POST /api/v1/models/<id>/train   run synthetic training workflow (orchestrator "few_shot")
 - POST /api/v1/models/<id>/test    run test result ingestion (store ModelLabel + InferencePeriod + EvaluationRun)
"""
from flask_restful import Resource
from flask import request, jsonify
from marshmallow import EXCLUDE
import json
import traceback

from src import storage
from src.storage.ai_models import AIModel
from src.storage.labels import Label
from src.storage.models_labels import ModelLabel
from src.storage.inference_periods import InferencePeriod
from src.storage.outputs import Output

# optional EvaluationRun
try:
    from src.storage.evaluation_runs import EvaluationRun
except Exception:
    EvaluationRun = None

from ..serializers.labels import LabelSchema

# orchestrator import
try:
    from src.pipeline.orchestrator import orchestrate
    from src.synthimage.generator import generate_images
except Exception:
    orchestrate = None
    generate_images = None

label_schema = LabelSchema(unknown=EXCLUDE)


class ModelsList(Resource):
    """List and create AI models."""

    def get(self):
        """
        Get all AI models
        ---
        tags:
          - Models
        responses:
          200:
            description: list of ai models
        """
        sess = storage.database.session
        all_models = sess.query(AIModel).order_by(AIModel.created_at.desc()).all()
        out = [{"id": m.id, "name": m.name, "description": getattr(m, "description", None)} for m in all_models]
        return jsonify(out), 200

    def post(self):
        """
        Create a new AIModel
        ---
        tags:
          - Models
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                name:
                  type: string
                description:
                  type: string
        responses:
          201:
            description: model created
          422:
            description: validation error
        """
        raw = request.get_json(force=True, silent=True)
        if not raw or not raw.get("name"):
            return {"message": "name is required"}, 422
        sess = storage.database.session
        # ensure unique
        exist = sess.query(AIModel).filter_by(name=raw["name"]).one_or_none()
        if exist:
            return {"message": "ai model with this name already exists", "id": exist.id}, 409
        m = AIModel(name=raw["name"], description=raw.get("description"))
        sess.add(m)
        sess.commit()
        return {"id": m.id, "name": m.name, "description": m.description}, 201


class ModelSingle(Resource):
    """Get / Update / Delete single AI model."""

    def get(self, model_id):
        """
        Get AI model by id
        ---
        tags:
          - Models
        parameters:
          - in: path
            name: model_id
            required: true
            type: string
        responses:
          200: {}
          404: {}
        """
        sess = storage.database.session
        m = sess.query(AIModel).get(model_id)
        if not m:
            return {"message": "not found"}, 404
        return {"id": m.id, "name": m.name, "description": m.description}, 200

    def put(self, model_id):
        """
        Update AI model
        ---
        tags:
          - Models
        parameters:
          - in: path
            name: model_id
            required: true
            type: string
          - in: body
            name: body
            required: false
            schema:
              type: object
              properties:
                name: { type: string }
                description: { type: string }
        responses:
          200: updated
          404: not found
        """
        raw = request.get_json(force=True, silent=True) or {}
        sess = storage.database.session
        m = sess.query(AIModel).get(model_id)
        if not m:
            return {"message": "not found"}, 404
        if raw.get("name"):
            m.name = raw["name"]
        if raw.get("description") is not None:
            m.description = raw["description"]
        sess.commit()
        return {"id": m.id, "name": m.name, "description": m.description}, 200

    def delete(self, model_id):
        """
        Delete AI model
        ---
        tags:
          - Models
        parameters:
          - in: path
            name: model_id
            required: true
            type: string
        responses:
          204: deleted
        """
        sess = storage.database.session
        m = sess.query(AIModel).get(model_id)
        if not m:
            return {"message": "not found"}, 404
        sess.delete(m)
        sess.commit()
        return {}, 204


class ModelTrain(Resource):
    """Trigger synthetic training/evaluation via orchestrator (few_shot/zero_shot)."""

    def post(self, model_id):
        """
        Start a training run (synthetic few_shot by default).
        Request JSON may include:
          - mode: "few_shot" | "zero_shot"
          - labels: [ "car", "bicycle", ... ]    # required for synthetic flows
          - n_per_label_train: int
          - n_per_label_test: int
          - use_existing_classifier: bool
          - classifier_params: dict
          - gen_kwargs: dict  (forwarded to generate_images)
        The endpoint will create an EvaluationRun (if model supports it) and persist ModelLabel
        and InferencePeriod rows returned by the orchestrator.
        ---
        tags:
          - Models
        consumes:
          - application/json
        produces:
          - application/json
        responses:
          201:
            description: Training run recorded (run_id and counts)
          400:
            description: invalid request
          404:
            description: model not found
        """
        raw = request.get_json(force=True, silent=True) or {}
        sess = storage.database.session
        m = sess.query(AIModel).get(model_id)
        if not m:
            return {"message": "model not found"}, 404

        mode = raw.get("mode", "few_shot")
        labels = raw.get("labels")
        if mode in ("few_shot",) and (not labels or not isinstance(labels, list)):
            return {"message": "labels list required for synthetic few_shot training"}, 400

        # if orchestrator is unavailable, return error
        if orchestrate is None:
            return {"message": "orchestrator not available"}, 500

        # call orchestrator for synthetic flows
        try:
            if mode == "few_shot":
                res = orchestrate(
                    mode="few_shot",
                    generate_images_fn=generate_images,
                    labels=labels,
                    n_per_label_train=raw.get("n_per_label_train", 10),
                    n_per_label_test=raw.get("n_per_label_test", 5),
                    store_root=raw.get("store_root"),
                    classifier_path=raw.get("classifier_path"),
                    gen_kwargs=raw.get("gen_kwargs"),
                    use_existing_classifier=raw.get("use_existing_classifier", False),
                    classifier_params=raw.get("classifier_params"),
                    few_shot_kwargs=raw.get("few_shot_kwargs"),
                    verbose=False
                )
            elif mode == "zero_shot":
                res = orchestrate(
                    mode="zero_shot",
                    generate_images_fn=generate_images,
                    labels=labels,
                    candidate_labels=raw.get("candidate_labels", labels),
                    n_per_label_test=raw.get("n_per_label_test", 5),
                    store_root=raw.get("store_root"),
                    gen_kwargs=raw.get("gen_kwargs"),
                    zero_shot_kwargs=raw.get("zero_shot_kwargs"),
                    verbose=False
                )
            else:
                return {"message": f"unsupported mode: {mode}"}, 400

            # Expected res contains keys like:
            # res["run_id"], res["label_metrics"] (list of {label, accuracy, precision, recall, f1_score}),
            # res["latencies"] (list of floats or dicts)
            run_id = res.get("run_id") or None

            # Create EvaluationRun if model has such model and we can supply id
            created_run = None
            if EvaluationRun:
                created_run = EvaluationRun(ai_model_id=m.id, run_type="train", metadata=json.dumps(res.get("metadata", {})) )
                sess.add(created_run)
                sess.flush()
                # If orchestrator supplied run_id, we can prefer that id (try to set it)
                if run_id:
                    try:
                        setattr(created_run, "id", run_id)
                    except Exception:
                        pass
                run_db_id = getattr(created_run, "id", None)
            else:
                run_db_id = run_id

            # Persist ModelLabel rows if provided
            model_label_rows = res.get("label_metrics") or res.get("model_labels") or []
            ml_objs = []
            for r in model_label_rows:
                # resolve label id creating label if needed
                label_name = r.get("label") or r.get("label_name")
                if not label_name:
                    continue
                # find or create label
                lbl = sess.query(Label).filter_by(name=label_name).one_or_none()
                if not lbl:
                    lbl = Label(name=label_name, description="auto-created by training")
                    sess.add(lbl)
                    sess.flush()
                ml = ModelLabel(ai_model_id=m.id,
                                label_id=lbl.id,
                                accuracy=float(r.get("accuracy", 0.0)),
                                precision=float(r.get("precision", 0.0)),
                                recall=float(r.get("recall", 0.0)),
                                f1_score=float(r.get("f1_score", 0.0)))
                if hasattr(ml, "run_id") and run_db_id:
                    setattr(ml, "run_id", run_db_id)
                ml_objs.append(ml)
            if ml_objs:
                sess.add_all(ml_objs)

            # Persist latencies if provided
            lat_rows = res.get("latencies") or res.get("inference_times") or []
            lat_objs = []
            for lat in lat_rows:
                if isinstance(lat, dict):
                    input_id = lat.get("input_id")
                    val = lat.get("value")
                else:
                    input_id = None
                    val = float(lat)
                ip = InferencePeriod(ai_model_id=m.id, input_id=input_id, value=float(val))
                if hasattr(ip, "run_id") and run_db_id:
                    setattr(ip, "run_id", run_db_id)
                lat_objs.append(ip)
            if lat_objs:
                sess.add_all(lat_objs)

            sess.commit()
            return {"run_id": run_db_id, "model_labels_inserted": len(ml_objs), "latencies_inserted": len(lat_objs)}, 201

        except Exception as e:
            sess.rollback()
            traceback.print_exc()
            return {"message": "training failed", "error": str(e)}, 500


class ModelTest(Resource):
    """Accept test results (external) and persist them to DB as ModelLabel/InferencePeriod rows."""

    def post(self, model_id):
        """
        Ingest test results:
        {
          "dataset": "testset",
          "results": [{"label_id": "<id>", "accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1_score": 0.825}, ...],
          "latencies": [{"input_id": "<id>", "value": 0.12}, ...],
          "metadata": {...}
        }
        ---
        tags:
          - Models
        consumes:
          - application/json
        responses:
          201:
            description: results stored
          422:
            description: validation error
          404:
            description: model not found
        """
        raw = request.get_json(force=True, silent=True) or {}
        sess = storage.database.session
        m = sess.query(AIModel).get(model_id)
        if not m:
            return {"message": "model not found"}, 404

        results = raw.get("results") or []
        latencies = raw.get("latencies") or []

        # create EvaluationRun record
        run_db = None
        if EvaluationRun:
            run_db = EvaluationRun(ai_model_id=m.id, run_type="test", metadata=json.dumps(raw.get("metadata", {})))
            sess.add(run_db)
            sess.flush()
            run_id_db = getattr(run_db, "id", None)
        else:
            run_id_db = None

        ml_objs = []
        for r in results:
            lid = r.get("label_id")
            if not lid:
                # try label name
                if r.get("label"):
                    lbl = sess.query(Label).filter_by(name=r["label"]).one_or_none()
                    if lbl:
                        lid = lbl.id
                    else:
                        # create new label
                        new_lbl = Label(name=r["label"], description="created during test ingestion")
                        sess.add(new_lbl)
                        sess.flush()
                        lid = new_lbl.id
            # ensure label exists
            lbl_obj = sess.query(Label).get(lid)
            if not lbl_obj:
                continue
            ml = ModelLabel(ai_model_id=m.id,
                            label_id=lid,
                            accuracy=float(r.get("accuracy", 0.0)),
                            precision=float(r.get("precision", 0.0)),
                            recall=float(r.get("recall", 0.0)),
                            f1_score=float(r.get("f1_score", 0.0)))
            if hasattr(ml, "run_id") and run_id_db:
                setattr(ml, "run_id", run_id_db)
            ml_objs.append(ml)
        if ml_objs:
            sess.add_all(ml_objs)

        ip_objs = []
        for lat in latencies:
            input_id = lat.get("input_id")
            val = lat.get("value")
            if val is None:
                continue
            ip = InferencePeriod(ai_model_id=m.id, input_id=input_id, value=float(val))
            if hasattr(ip, "run_id") and run_id_db:
                setattr(ip, "run_id", run_id_db)
            ip_objs.append(ip)
        if ip_objs:
            sess.add_all(ip_objs)

        sess.commit()
        return {"run_id": getattr(run_db, "id", None), "model_labels_inserted": len(ml_objs), "latencies_inserted": len(ip_objs)}, 201
