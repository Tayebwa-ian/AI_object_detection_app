#!/usr/bin/python3
"""
Input Views module - uses image storage helper and persists aggregated prediction results.

Endpoints:
 - GET  /api/v1/inputs           List inputs (paginated)
 - POST /api/v1/inputs           Upload image, create Input, run classification (optional) and persist aggregated results
 - GET  /api/v1/inputs/<id>      Retrieve single input
 - DELETE /api/v1/inputs/<id>    Delete input and its stored image

Behavior notes:
 - This view uses `src.utils.images.upload_image(request)` to store uploaded files.
 - Aggregated results are persisted from the normalized structure returned by
   `_construct_prediction_response()` (based on your pipeline's "result" field).
 - Per-stage models (segmentation / feature / classifier) are ensured in the AIModel table.
 - Inference times are stored in InferencePeriod rows linked to their AIModel.
 - Outputs store aggregated counts per label and `confidence` set to average segment confidence.
"""

from flask_restful import Resource
from flask import request, jsonify, make_response
from marshmallow import EXCLUDE
import traceback
import json
import os
import clip  # optional; orchestrator may require clip model objects

from src import storage
from src.storage.inputs import Input
from src.storage.labels import Label
from src.storage.outputs import Output
from src.storage.ai_models import AIModel
from src.storage.inference_periods import InferencePeriod
from src.api.utils.image_utils import upload_image, delete_image  # uses your image storage script
from ..serializers.inputs import InputSchema

# Try to import orchestrator (may be absent in unit tests)
try:
    from src.pipeline.orchestrator import orchestrate
except Exception:
    orchestrate = None

# Optional EvaluationRun model (if present in your schema)
try:
    from src.storage.evaluation_runs import EvaluationRun
except Exception:
    EvaluationRun = None


# -----------------------------
# Configuration & Schemas
# -----------------------------
DISALLOWED_TERMS = {
    "military", "weapon", "tank", "gun",
    "firearm", "explosives", "explosive", "military vehicle"
}
input_schema = InputSchema(unknown=EXCLUDE)
inputs_schema = InputSchema(many=True)


# -----------------------------
# Helper functions
# -----------------------------
def _contains_disallowed(s: str) -> bool:
    """
    Return True if the string contains any disallowed safety terms.
    """
    if not s:
        return False
    low = s.lower()
    for t in DISALLOWED_TERMS:
        if t in low:
            return True
    return False


def _get_or_create_model(sess, model_name: str):
    """
    Ensure an AIModel row exists for model_name.
    Returns the AIModel.id or None if model_name falsy.
    """
    if not model_name:
        return None
    mdl = sess.query(AIModel).filter_by(name=model_name).one_or_none()
    if mdl:
        return mdl.id
    new_m = AIModel(name=model_name, description="auto-created from pipeline")
    sess.add(new_m)
    sess.flush()
    return new_m.id


def _get_or_create_label(sess, label_name: str):
    """
    Ensure a Label row exists for label_name. Create it if missing and safe.
    Returns Label.id or None if disallowed/falsy.
    """
    if not label_name:
        return None
    if _contains_disallowed(label_name):
        return None
    lbl = sess.query(Label).filter_by(name=label_name).one_or_none()
    if lbl:
        return lbl.id
    new_lbl = Label(name=str(label_name)[:128], description="auto-created from pipeline")
    sess.add(new_lbl)
    sess.flush()
    return new_lbl.id


def _construct_prediction_response(raw: dict) -> dict:
    """
    Normalize the orchestrator pipeline raw response into a consistent structure.

    Expected raw (user prediction):
      {
        "run_id": "...",
        "mode": "user_few_shot",
        "result": {
          "segments": [ {...} ],
          "counts": {"cat": 4, "car": 1},
          "metadata": {
            "segmentation_model": "...",
            "feature_model": "...",
            "elapsed_time": 161.5,
            "avg_feature_time": 0.73,
            "avg_classifier_time": 0.0014
          }
        }
      }

    Returns:
      {
        "predictions": { label: count, ... },
        "segments": [ { "segment_path", "predicted_label", "prediction_confidence", "classifier_source", "feature_model" }, ... ],
        "run_id": ...,
        "mode": ...,
        "segmentation_model": ...,
        "feature_extractor_model": ...,
        "classifier_name": ...,
        "segmentation_inference_time": ...,
        "avg_feature_extractor_inference_time": ...,
        "avg_classifier_inference_time": ...,
        "average_confidence": ...
      }
    """
    result = raw.get("result", {}) or {}
    segments_raw = result.get("segments", []) or []

    segments = []
    feature_times = []
    confidences = []
    classifier_name = None

    for seg in segments_raw:
        feature_meta = seg.get("feature_meta") or {}
        classifier_pred = seg.get("classifier_pred") or {}

        # gather feature extractor inference_time (per-segment)
        if feature_meta.get("inference_time") is not None:
            try:
                feature_times.append(float(feature_meta["inference_time"]))
            except Exception:
                pass

        # gather classifier confidence
        if classifier_pred.get("score") is not None:
            try:
                confidences.append(float(classifier_pred["score"]))
            except Exception:
                pass

        # prefer the classifier source from the first segment that has it
        if not classifier_name and classifier_pred.get("source"):
            classifier_name = classifier_pred.get("source")

        segments.append({
            "segment_path": seg.get("segment_path"),
            "predicted_label": classifier_pred.get("label"),
            "prediction_confidence": classifier_pred.get("score"),
            "classifier_source": classifier_pred.get("source"),
            "feature_model": feature_meta.get("model_name")
        })

    metadata = result.get("metadata", {}) or {}

    avg_feature_time = float(sum(feature_times) / len(feature_times)) if feature_times else metadata.get("avg_feature_time", 0.0)
    avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0

    return {
        "predictions": result.get("counts", {}) or {},
        "segments": segments,
        "run_id": raw.get("run_id"),
        "mode": raw.get("mode"),
        "segmentation_model": metadata.get("segmentation_model"),
        "feature_extractor_model": metadata.get("feature_model"),
        "classifier_name": classifier_name,
        # segmentation total inference time as `elapsed_time` per your spec
        "segmentation_inference_time": metadata.get("elapsed_time", 0.0),
        "avg_feature_extractor_inference_time": avg_feature_time,
        "avg_classifier_inference_time": metadata.get("avg_classifier_time", 0.0),
        "average_confidence": avg_confidence
    }


def _persist_aggregated_results(sess, input_id: str, struct: dict):
    """
    Persist aggregated results into DB.

    - Ensure AIModel rows exist: segmentation, feature, classifier.
    - Persist Output rows (one per label count) with confidence = average segment confidence.
    - Persist InferencePeriod rows for segmentation, feature, classifier models.
    Returns (list_of_Output_objs, list_of_InferencePeriod_objs).
    """
    output_objs = []
    inference_objs = []

    # Ensure models exist
    seg_m_id = _get_or_create_model(sess, struct.get("segmentation_model"))
    feat_m_id = _get_or_create_model(sess, struct.get("feature_extractor_model"))
    clf_m_id = _get_or_create_model(sess, struct.get("classifier_name"))

    avg_conf = struct.get("average_confidence", 0.0)

    # Create Output rows for aggregated counts
    for label_name, count in (struct.get("predictions") or {}).items():
        lbl_id = _get_or_create_label(sess, label_name)
        if not lbl_id:
            # skip disallowed or invalid labels
            continue
        out = Output(
            input_id=input_id,
            label_id=lbl_id,
            ai_model_id=clf_m_id,  # classifier produced label counts
            predicted_count=int(count),
            confidence=float(avg_conf)
        )
        output_objs.append(out)

    if output_objs:
        sess.add_all(output_objs)

    # Persist inference times by stage (if present)
    seg_time = struct.get("segmentation_inference_time")
    feat_time = struct.get("avg_feature_extractor_inference_time")
    clf_time = struct.get("avg_classifier_inference_time")

    if seg_time and seg_m_id:
        ip = InferencePeriod(ai_model_id=seg_m_id, input_id=input_id, value=float(seg_time))
        if struct.get("run_id") and hasattr(ip, "run_id"):
            ip.run_id = struct.get("run_id")
        inference_objs.append(ip)

    if feat_time and feat_m_id:
        ipf = InferencePeriod(ai_model_id=feat_m_id, input_id=input_id, value=float(feat_time))
        if struct.get("run_id") and hasattr(ipf, "run_id"):
            ipf.run_id = struct.get("run_id")
        inference_objs.append(ipf)

    if clf_time and clf_m_id:
        ipc = InferencePeriod(ai_model_id=clf_m_id, input_id=input_id, value=float(clf_time))
        if struct.get("run_id") and hasattr(ipc, "run_id"):
            ipc.run_id = struct.get("run_id")
        inference_objs.append(ipc)

    if inference_objs:
        sess.add_all(inference_objs)

    return output_objs, inference_objs


# -----------------------------
# Resources
# -----------------------------
class InputList(Resource):
    """Resource for listing and creating Inputs."""

    def get(self):
        """
        Get a paginated list of inputs.
        ---
        tags:
          - Inputs
        parameters:
          - name: page
            in: query
            type: integer
            default: 1
          - name: per_page
            in: query
            type: integer
            default: 50
        responses:
          200:
            description: Paginated list of inputs
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
        Upload image, create Input, and optionally run the user classification pipeline.

        Accepts multipart/form-data fields:
          - image (file) required OR supply 'image_path' in JSON to use an existing path
          - prompt (string) optional
          - advanced_mode (boolean-like) optional -> run few-shot when true
          - is_test (boolean-like) optional -> mark input as test

        Returns:
          201: created input and aggregated predictions (if orchestrator run)
          400/422/500: error responses
        ---
        tags:
          - Inputs
        consumes:
          - multipart/form-data
        produces:
          - application/json
        """
        sess = storage.database.session

        # If user provided multipart/form-data with file, store it using your upload helper.
        image_filename = None
        # upload_image returns filename (string) or a Flask response (on error)
        if request.files and "image" in request.files:
            image_upload_result = upload_image(request)
            if not isinstance(image_upload_result, str):
                # upload_image returns a Flask response on error -> forward it
                return image_upload_result
            image_filename = image_upload_result
            # Recreate a full (filesystem) path to pass to orchestrator if needed.
            upload_folder = os.getenv("UPLOAD_FOLDER") or "media"
            full_image_path = os.path.join(upload_folder, image_filename)
        else:
            # If no file, allow client to pass image_path in JSON body (for internal/dev use)
            data_json = request.get_json(silent=True) or {}
            image_path_from_json = data_json.get("image_path")
            if image_path_from_json:
                image_filename = image_path_from_json
                full_image_path = image_filename
            else:
                return {"message": "No image uploaded and no image_path provided"}, 400

        # Read form fields if multipart, else JSON
        if request.form:
            prompt = request.form.get("prompt", "") or ""
            advanced_mode = request.form.get("advanced_mode", "false").lower() in ("1", "true", "yes")
            is_test_flag = request.form.get("is_test", "false").lower() in ("1", "true", "yes")
            candidate_labels = request.form.getlist("candidate_labels") or []
        else:
            data = request.get_json(silent=True) or {}
            prompt = data.get("prompt", "") or ""
            advanced_mode = bool(data.get("advanced_mode", False))
            is_test_flag = bool(data.get("is_test", False))
            candidate_labels = data.get("candidate_labels") or []

        # Safety check on prompt and candidate labels
        if _contains_disallowed(prompt):
            # cleanup uploaded file if we stored it in this request
            if image_filename and request.files and "image" in request.files:
                try:
                    delete_image(image_filename)
                except Exception:
                    pass
            return {"message": "prompt contains disallowed content"}, 422
        if any(_contains_disallowed(lb) for lb in candidate_labels):
            if image_filename and request.files and "image" in request.files:
                try:
                    delete_image(image_filename)
                except Exception:
                    pass
            return {"message": "candidate_labels contains disallowed terms"}, 422

        # Create Input DB record and persist image path
        new_input = Input(
            prompt=prompt,
            image_path=image_filename,
            is_few_shot=advanced_mode,
            is_zero_shot=(not advanced_mode),
            is_test=is_test_flag
        )
        sess.add(new_input)
        sess.flush()  # obtain new_input.id

        aggregated_struct = None
        created_outputs = []
        created_inferences = []

        # Run the orchestrator if available and requested (we run for both few_shot and zero_shot user modes)
        if orchestrate:
            try:
                # Attempt to load clip model if needed by orchestrator (safe try)
                try:
                    clip_model, clip_preprocess = clip.load(os.getenv("CLIP_MODEL", "ViT-B/32"), device=os.getenv("DEVICE", "cpu"))
                except Exception:
                    clip_model = clip_preprocess = None

                raw_res = orchestrate(
                    mode="user_few_shot" if advanced_mode else "user_zero_shot",
                    segmentation_model_name="sam",
                    feature_extractor_name="resnet",
                    labels=prompt,
                    candidate_labels=candidate_labels,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    clip_device=os.getenv("DEVICE", "cpu"),
                    image_path=full_image_path,
                )

                # Normalize pipeline response to our expected structure
                aggregated_struct = _construct_prediction_response(raw_res)

                # Persist aggregated results into DB: outputs + inference periods + models+labels
                created_outputs, created_inferences = _persist_aggregated_results(sess, new_input.id, aggregated_struct)

                # Optionally create an EvaluationRun record if you have that model and run id
                run_id = aggregated_struct.get("run_id")
                if run_id and EvaluationRun:
                    if not sess.query(EvaluationRun).get(run_id):
                        classifier_model_id = _get_or_create_model(sess, aggregated_struct.get("classifier_name"))
                        run = EvaluationRun(id=run_id, ai_model_id=classifier_model_id,
                                            run_type="user_classification", metadata=json.dumps(raw_res))
                        sess.add(run)

                # Commit everything for this input+predictions
                sess.commit()

            except Exception as exc:
                # Rollback and remove uploaded image (avoid leaving orphaned file)
                sess.rollback()
                if image_filename and request.files and "image" in request.files:
                    try:
                        delete_image(image_filename)
                    except Exception:
                        pass
                traceback.print_exc()
                return {"message": "classification failed", "error": str(exc)}, 500
        else:
            # No orchestrator: just commit the input record (image already saved)
            sess.commit()

        # Build response to client
        response = {"input": input_schema.dump(new_input)}
        if aggregated_struct:
            response["predictions"] = aggregated_struct
            response["created_outputs"] = len(created_outputs)
            response["created_inference_rows"] = len(created_inferences)

        return make_response(jsonify(response), 201)


class InputSingle(Resource):
    """Resource for retrieving and deleting single Input records."""

    def get(self, input_id):
        """
        Retrieve a single input by its ID.
        ---
        tags:
          - Inputs
        parameters:
          - name: input_id
            in: path
            type: string
            required: true
        responses:
          200:
            description: Input record
          404:
            description: Not found
        """
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "not found"}, 404
        return input_schema.dump(obj), 200

    def delete(self, input_id):
        """
        Delete an input and its stored image (if any).
        ---
        tags:
          - Inputs
        parameters:
          - name: input_id
            in: path
            type: string
            required: true
        responses:
          204:
            description: Deleted successfully
          404:
            description: Not found
        """
        sess = storage.database.session
        obj = sess.query(Input).get(input_id)
        if not obj:
            return {"message": "not found"}, 404

        # Attempt to delete the stored image file (ignore errors)
        if obj.image_path:
            try:
                delete_image(obj.image_path)
            except Exception:
                traceback.print_exc()

        # Delete the DB record (cascade should remove related Outputs/InferencePeriods if configured)
        sess.delete(obj)
        sess.commit()
        return {}, 204
