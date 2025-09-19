#!/usr/bin/python3
"""
Multi-Stage AI Training Views

Endpoints:
 - POST /api/v1/models/train -> Trigger multi-stage few-shot / zero-shot workflow
"""
from flask_restful import Resource
from flask import request, jsonify
import json
import traceback

from src import storage
from src.storage.ai_models import AIModel
from src.storage.labels import Label
from src.storage.models_labels import ModelLabel
from src.storage.inference_periods import InferencePeriod

try:
    from src.storage.evaluation_runs import EvaluationRun
except Exception:
    EvaluationRun = None

try:
    from src.pipeline.orchestrator import orchestrate
    from src.synthimage.generator import generate_images
except Exception:
    orchestrate = None
    generate_images = None


class MultiStageTrain(Resource):
    """
    Trigger multi-stage AI training workflow: segmentation -> feature extraction -> classification,
    persist results, metrics, and evaluation run.

    ---
    tags:
      - Models
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - mode
              - labels
              - n_per_label_train
              - n_per_label_test
              - models
            properties:
              mode:
                type: string
                enum: [few_shot, zero_shot]
                description: "Training mode"
                example: "few_shot"
              labels:
                type: array
                description: "List of labels to train/test on"
                items:
                  type: string
                example: ["cat", "dog", "person"]
              n_per_label_train:
                type: integer
                description: "Number of training samples per label"
                example: 10
              n_per_label_test:
                type: integer
                description: "Number of testing samples per label"
                example: 5
              models:
                type: object
                description: "Models to use at each stage"
                required:
                  - segmentation
                  - feature_extraction
                  - classification
                properties:
                  segmentation:
                    type: string
                    description: "Model ID for segmentation stage"
                    example: "seg_model_uuid"
                  feature_extraction:
                    type: string
                    description: "Model ID for feature extraction stage"
                    example: "feat_model_uuid"
                  classification:
                    type: string
                    description: "Model ID for classification stage"
                    example: "clf_model_uuid"
    responses:
      201:
        description: Multi-stage training run recorded
        content:
          application/json:
            schema:
              type: object
              properties:
                run_id:
                  type: string
                  example: "123e4567-e89b-12d3-a456-426614174000"
                labels_used:
                  type: array
                  items:
                    type: string
                  example: ["cat", "dog", "person"]
                model_labels_inserted:
                  type: integer
                  example: 3
                inference_periods_inserted:
                  type: integer
                  example: 1
      400:
        description: Invalid request
      500:
        description: Training failed
    """

    def post(self):
        raw = request.get_json(force=True, silent=True) or {}
        sess = storage.database.session

        # Required fields
        mode = raw.get("mode", "few_shot")
        submitted_labels = raw.get("labels", [])
        n_train = int(raw.get("n_per_label_train", 10))
        n_test = int(raw.get("n_per_label_test", 5))
        model_ids = raw.get("models", {})

        if mode not in ["few_shot", "zero_shot"]:
            return {"message": "Invalid mode"}, 400
        if not submitted_labels or not isinstance(submitted_labels, list):
            return {"message": "labels list required"}, 400
        if not all(stage in model_ids for stage in ["segmentation", "feature_extraction", "classification"]):
            return {"message": "models object must contain segmentation, feature_extraction, classification"}, 400
        if orchestrate is None:
            return {"message": "Orchestrator not available"}, 500

        try:
            # Merge and store labels
            final_labels = self._merge_and_store_labels(sess, submitted_labels)

            # Resolve model IDs to names
            stage_models = self._resolve_models(sess, model_ids)

            # Run orchestrator
            orchestrator_res = self._run_orchestrator(mode, final_labels, n_train, n_test, stage_models)

            # Persist EvaluationRun
            run_db = self._create_evaluation_run(sess, orchestrator_res)

            # Persist ModelLabel metrics
            ml_objs = self._store_model_labels(sess, run_db, orchestrator_res)

            # Persist InferencePeriod metrics
            ip_objs = self._store_inference_period(sess, run_db, orchestrator_res)

            sess.commit()
            return {
                "run_id": getattr(run_db, "id", None),
                "labels_used": final_labels,
                "model_labels_inserted": len(ml_objs),
                "inference_periods_inserted": len(ip_objs)
            }, 201

        except Exception as e:
            sess.rollback()
            traceback.print_exc()
            return {"message": "Training failed", "error": str(e)}, 500

    # -------------------- Helper Functions -------------------- #

    def _merge_and_store_labels(self, sess, submitted_labels):
        """Merge DB labels with user-submitted labels, insert missing labels."""
        submitted_labels = [lbl.strip().lower() for lbl in submitted_labels if lbl]
        db_labels = [lbl.name.lower() for lbl in sess.query(Label).all()]
        merged_labels = sorted(set(submitted_labels) | set(db_labels))

        for lbl in merged_labels:
            if lbl not in db_labels:
                sess.add(Label(name=lbl, description="auto-created at training"))
        sess.flush()
        return merged_labels

    def _resolve_models(self, sess, model_ids):
        """Retrieve model names from DB given stage model IDs."""
        stage_models = {}
        for stage, mid in model_ids.items():
            model = sess.query(AIModel).get(mid)
            if not model:
                raise ValueError(f"Model with ID {mid} not found for stage {stage}")
            stage_models[stage] = model.name
        return stage_models

    def _run_orchestrator(self, mode, labels, n_train, n_test, stage_models):
        """Call orchestrator with resolved model names per stage."""
        return orchestrate(
            mode=mode,
            segmentation_model_name=stage_models.get("segmentation"),
            feature_extractor_name=stage_models.get("feature_extraction"),
            few_shot_classifier_type="logistic",
            labels=labels,
            n_per_label_train=n_train,
            n_per_label_test=n_test,
            candidate_labels=labels,  # for zero-shot
            verbose=True
        )

    def _create_evaluation_run(self, sess, orchestrator_res):
        """Persist EvaluationRun row with metadata."""
        if not EvaluationRun:
            return None
        metadata = {
            "classifier_info": orchestrator_res["result"].get("classifier_info"),
            "train_summary": orchestrator_res["result"].get("summary", {}).get("train_summary"),
            "metrics": orchestrator_res["result"].get("metrics"),
            "avg_inference_time": orchestrator_res["result"].get("avg_inference_time"),
            "total_elapsed_time": orchestrator_res["result"].get("total_elapsed_time")
        }
        run = EvaluationRun(ai_model_id=None, run_type="train", metadata=json.dumps(metadata))
        sess.add(run)
        sess.flush()
        return run

    def _store_model_labels(self, sess, run_db, orchestrator_res):
        """Persist ModelLabel metrics per label."""
        ml_objs = []
        per_label_metrics = orchestrator_res["result"].get("metrics", {}).get("per_label", {})
        for label_name, metrics in per_label_metrics.items():
            if label_name.lower() in ["macro avg", "weighted avg"]:
                continue
            lbl = sess.query(Label).filter(Label.name.ilike(label_name)).one_or_none()
            if not lbl:
                lbl = Label(name=label_name.lower(), description="auto-created by training")
                sess.add(lbl)
                sess.flush()
            ml = ModelLabel(
                ai_model_id=None,
                label_id=lbl.id,
                precision=float(metrics.get("precision", 0.0)),
                recall=float(metrics.get("recall", 0.0)),
                f1_score=float(metrics.get("f1", 0.0)),
                run_id=getattr(run_db, "id", None)
            )
            ml_objs.append(ml)
        if ml_objs:
            sess.add_all(ml_objs)
        return ml_objs

    def _store_inference_period(self, sess, run_db, orchestrator_res):
        """Persist average inference time from orchestrator."""
        avg_time = orchestrator_res["result"].get("avg_inference_time")
        if avg_time is None:
            return []
        ip = InferencePeriod(ai_model_id=None, value=float(avg_time), run_id=getattr(run_db, "id", None))
        sess.add(ip)
        return [ip]
