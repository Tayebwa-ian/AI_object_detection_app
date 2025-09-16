#!/usr/bin/python3
"""
Input Views module - enhanced to support orchestration (user few/zero shot classification).

This module exposes:
 - GET  /api/v1/inputs           List inputs (paginated)
 - POST /api/v1/inputs           Create an input and optionally run immediate user classification
 - GET  /api/v1/inputs/<id>      Retrieve single input
 - PUT  /api/v1/inputs/<id>      Update input record
 - DELETE /api/v1/inputs/<id>    Delete input

When a POST includes a "mode" field with value "user_few_shot" or "user_zero_shot",
this view will:
 - create an Input row,
 - call src.orchestrator.orchestrate(...) to classify the provided image,
 - persist Outputs and InferencePeriod rows returned by orchestrator (resolving/creating labels if needed),
 - attach any run_id or model metrics if available.
"""
from flask_restful import Resource
from flask import request, jsonify, make_response
from marshmallow import ValidationError, EXCLUDE
import json
import traceback

from src import storage
from src.storage.inputs import Input
from src.storage.labels import Label
from src.storage.outputs import Output
from src.storage.ai_models import AIModel
from src.storage.models_labels import ModelLabel
from src.storage.inference_periods import InferencePeriod

# Optional evaluation run model (may not exist in older schemas)
try:
    from src.storage.evaluation_runs import EvaluationRun
except Exception:
    EvaluationRun = None  # optional

from ..serializers.inputs import InputSchema

# Attempt to import orchestrator
try:
    from src.pipeline.orchestrator import orchestrate
except Exception:
    orchestrate = None

# Basic blacklist for safety checks
DISALLOWED_TERMS = {"military", "weapon", "tank", "gun", "firearm", "explosives", "explosive", "military vehicle"}

input_schema = InputSchema(unknown=EXCLUDE)
inputs_schema = InputSchema(many=True)


def _contains_disallowed(s: str) -> bool:
    """Return True if s contains any disallowed terms."""
    if not s:
        return False
    low = s.lower()
    for term in DISALLOWED_TERMS:
        if term in low:
            return True
    return False


class InputList(Resource):
    """Handles requests for multiple Inputs."""

    def get(self):
        """
        Get paginated inputs
        ---
        tags:
          - Inputs
        parameters:
          - in: query
            name: page
            type: integer
          - in: query
            name: per_page
            type: integer
        responses:
          200:
            description: list of inputs (paginated)
        """
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        sess = storage.database.session

        q = sess.query(Input).order_by(Input.created_at.desc())
        total = q.count()
        items = q.offset((page - 1) * per_page).limit(per_page).all()

        return make_response(jsonify({
            "page": page,
            "per_page": per_page,
            "total": total,
            "items": inputs_schema.dump(items)
        }), 200)

    def post(self):
        """
        Create a new input. Optionally run immediate user classification.

        Example JSON (simple create):
        {
            "prompt": "cars", # label to predict
            "image_path": "/tmp/img.jpg",
            "advanced_mode": false # if true run few shot otherwise run zero shot
            "candidate_labels": ["bicycle","car"],  # for zero_shot
        }

        ---
        tags:
          - Inputs
        consumes:
          - application/json
        produces:
          - application/json
        responses:
          201:
            description: input created (and classification results if requested)
          422:
            description: validation / safety error
        """
        data = request.get_json(force=True, silent=True)
        if not data:
            return {"message": "Invalid or missing JSON body"}, 400

        # Safety check on prompt (also candidate_labels below)
        prompt = data.get("prompt") or ""
        if _contains_disallowed(prompt):
            return {"message": "prompt contains disallowed content"}, 422

        # Create Input row first (so we have an id)
        sess = storage.database.session
        new_input = Input()

        # add is_few_shot and is_zero_shot to data object
        if data["advanced_mode"]:
            data["is_few_shot"] = True
            data["is_zero_shot"] = False
        else:
            data["is_few_shot"] = False
            data["is_zero_shot"] = True

        # TODO we need to create a unique image path and store the image on the file system
        image_path = None

        # fill allowed fields from data
        for f in ("prompt", "image_path", "is_few_shot", "is_zero_shot"):
            if f in data:
                setattr(new_input, f, data[f])
        # perform data validation
        new_input = input_schema.load(new_input)
        sess.add(new_input)
        sess.flush()  # obtain id

        # If caller wants immediate classification/run
        mode = data.get("advanced_mode")
        predictions = []
        persisted_latencies = []
        run_info = None

        if mode:
            if orchestrate is None:
                sess.rollback()
                return {"message": "orchestrator is not available on the server"}, 500

            try:
                # Prepare orchestrator args
                candidate_labels = data.get("candidate_labels")

                # If mode is zero-shot and no candidate_labels provided, attempt to use DB labels
                if not mode and not candidate_labels:
                    # fetch label names from DB
                    candidate_labels = [l.name for l in sess.query(Label).with_entities(Label.name).all()]

                # safety check candidate labels
                if candidate_labels and any(_contains_disallowed(lb) for lb in candidate_labels):
                    sess.rollback()
                    return {"message": "candidate_labels contains disallowed terms"}, 422

                # call orchestrator - user modes
                if mode: # run few shot if true
                    res = orchestrate(
                        mode="user_few_shot",
                        segmentation_model_name="sam",
                        feature_extractor_name="resnet",
                        few_shot_classifier_type="logistic",
                        image_path=image_path,
                    )
                elif not mode: # run zero shot if true
                    res = orchestrate(
                        mode="user_zero_shot",
                        image_path=new_input.image_path,
                        candidate_labels=candidate_labels or [],
                        prototypes_store=data.get("prototypes_store"),
                        use_clip=data.get("use_clip", True),
                        use_prototypes=data.get("use_prototypes", True),
                        zero_shot_kwargs=data.get("zero_shot_kwargs"),
                        verbose=False
                    )

                # res is expected to be a dict. We support a few common shapes:
                # - res["outputs"] or res["predictions"] -> list of { "label": <name>|<id>, "predicted_count": int, "confidence": float, "bbox": optional }
                # - res["latencies"] -> list of { "value": float, "ai_model_id": optional }
                # - res["ai_model_id"] or res["ai_model_name"] or res["ai_model"] describing the model used
                # - res["run_id"] optional run identifier

                # Persist outputs
                preds = res.get("outputs") or res.get("predictions") or []
                # Helper: resolve label name -> id (create label if name not found)
                def _resolve_label_id(label_ref):
                    # label_ref might be an id (string) or a name
                    if not label_ref:
                        return None
                    # try by id
                    lbl = sess.query(Label).get(label_ref) if isinstance(label_ref, str) and len(label_ref) > 8 else None
                    if lbl:
                        return lbl.id
                    # by name
                    lbl = sess.query(Label).filter_by(name=label_ref).one_or_none()
                    if lbl:
                        return lbl.id
                    # create new Label if looks safe
                    if _contains_disallowed(label_ref):
                        return None
                    new_lbl = Label()
                    new_lbl.name = str(label_ref)[:128]
                    new_lbl.description = "auto-created from user classification"
                    sess.add(new_lbl)
                    sess.flush()
                    return new_lbl.id

                ai_model_id = None
                # try to find model id from res
                if res.get("ai_model_id"):
                    ai_model_id = res.get("ai_model_id")
                elif res.get("ai_model"):
                    # ai_model may be object or id or name
                    am = res.get("ai_model")
                    if isinstance(am, dict):
                        ai_model_id = am.get("id")
                        # if dict includes name but no id, create AIModel record
                        if not ai_model_id and am.get("name"):
                            existing = sess.query(AIModel).filter_by(name=am["name"]).one_or_none()
                            if existing:
                                ai_model_id = existing.id
                            else:
                                nm = AIModel(name=am["name"], description=am.get("description"))
                                sess.add(nm)
                                sess.flush()
                                ai_model_id = nm.id
                    elif isinstance(am, str):
                        # treat as id or name
                        found = sess.query(AIModel).get(am)
                        if found:
                            ai_model_id = found.id
                        else:
                            existing = sess.query(AIModel).filter_by(name=am).one_or_none()
                            if existing:
                                ai_model_id = existing.id

                # Build Output objects
                output_objs = []
                for p in preds:
                    label_ref = p.get("label") or p.get("label_name") or p.get("label_id")
                    label_id = _resolve_label_id(label_ref)
                    if label_ref and label_id is None:
                        # disallowed label
                        continue
                    oc = Output(
                        input_id=new_input.id,
                        label_id=label_id,
                        ai_model_id=ai_model_id,
                        predicted_count=int(p.get("predicted_count") or p.get("count") or 0),
                        confidence=float(p.get("confidence") or p.get("score") or 0.0)
                    )
                    # optional bbox
                    if p.get("bbox"):
                        try:
                            oc.bbox = json.dumps(p.get("bbox"))
                        except Exception:
                            oc.bbox = str(p.get("bbox"))
                    output_objs.append(oc)

                if output_objs:
                    sess.add_all(output_objs)

                # Persist latencies if provided
                lat_rows = res.get("latencies") or res.get("inference_times") or []
                lat_objs = []
                for lat in lat_rows:
                    val = lat.get("value") if isinstance(lat, dict) else (float(lat) if lat is not None else None)
                    if val is None:
                        continue
                    ip = InferencePeriod(ai_model_id=lat.get("ai_model_id") or ai_model_id,
                                         input_id=new_input.id,
                                         value=float(val))
                    # attach run id if available and attribute present
                    if EvaluationRun and hasattr(ip, "run_id") and res.get("run_id"):
                        setattr(ip, "run_id", res.get("run_id"))
                    lat_objs.append(ip)
                if lat_objs:
                    sess.add_all(lat_objs)

                # Persist model-label metrics if orchestrator returns per-label metrics (rare for user_* modes but possible)
                ml_rows = res.get("label_metrics") or res.get("per_label_metrics") or []
                ml_objs = []
                for mtr in ml_rows:
                    lid = _resolve_label_id(mtr.get("label") or mtr.get("label_id"))
                    if not lid:
                        continue
                    ml = ModelLabel(ai_model_id=ai_model_id,
                                    label_id=lid,
                                    accuracy=float(mtr.get("accuracy") or 0.0),
                                    precision=float(mtr.get("precision") or 0.0),
                                    recall=float(mtr.get("recall") or 0.0),
                                    f1_score=float(mtr.get("f1_score") or 0.0))
                    # attach run_id if supported
                    if hasattr(ml, "run_id") and res.get("run_id"):
                        setattr(ml, "run_id", res.get("run_id"))
                    ml_objs.append(ml)
                if ml_objs:
                    sess.add_all(ml_objs)

                # If the orchestrator returned a structured run (id, metadata) and we have EvaluationRun model, create it
                if res.get("run_id") and EvaluationRun:
                    # not strictly necessary if orchestrator already created run server-side, but we try
                    r_exist = sess.query(EvaluationRun).get(res.get("run_id"))
                    if not r_exist:
                        run = EvaluationRun(id=res.get("run_id"), ai_model_id=ai_model_id,
                                            run_type="user_classification", metadata=json.dumps(res.get("metadata") or {}))
                        sess.add(run)

                # finalize
                sess.commit()
                predictions = [dict(label=p.get("label"), predicted_count=p.get("predicted_count"), confidence=p.get("confidence")) for p in preds]
                persisted_latencies = [{"value": getattr(o, "value", None)} for o in lat_objs]

                run_info = {"run_id": res.get("run_id")} if res.get("run_id") else None

            except Exception as e:
                sess.rollback()
                traceback.print_exc()
                return {"message": "classification failed", "error": str(e)}, 500

        else:
            # No orchestration requested; just commit the Input
            sess.commit()

        # Build response
        resp = {"input": input_schema.dump(new_input)}
        if predictions:
            resp["predictions"] = predictions
        if persisted_latencies:
            resp["latencies"] = persisted_latencies
        if run_info:
            resp["run"] = run_info
        return make_response(jsonify(resp), 201)


class InputSingle(Resource):
    """Single Input operations (get, put, delete)."""

    def get(self, input_id):
        """
        Get a single input
        ---
        tags:
          - Inputs
        parameters:
          - in: path
            name: input_id
            type: string
            required: true
        responses:
          200:
            description: Input object
          404:
            description: not found
        """
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "not found"}, 404
        return input_schema.dump(obj), 200

    def put(self, input_id):
        """
        Update an input record
        ---
        tags:
          - Inputs
        parameters:
          - in: path
            name: input_id
            required: true
            type: string
          - in: body
            name: body
            required: true
            schema:
              $ref: '#/definitions/Input'
        responses:
          200:
            description: updated
        """
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "not found"}, 404

        data = request.get_json(force=True, silent=True)
        if not data:
            return {"message": "invalid JSON"}, 400

        try:
            data = input_schema.load(data, partial=True)
        except ValidationError as e:
            return {"message": "validation error", "errors": e.messages}, 422

        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        sess.commit()
        return input_schema.dump(obj), 200

    def delete(self, input_id):
        """
        Delete input
        ---
        tags:
          - Inputs
        parameters:
          - in: path
            name: input_id
            required: true
            type: string
        responses:
          204:
            description: deleted
        """
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "not found"}, 404
        sess.delete(obj)
        sess.commit()
        return {}, 204
