#!/usr/bin/python3
"""
Monitoring endpoints for AI model metrics and aggregations.

Endpoints:
- /metrics                -> Prometheus scrape (text format)
- /api/v1/metrics/summary -> JSON summary of metrics per model/label

Prometheus metrics exported:
- label_model_metric
- label_model_avg_confidence
- label_model_avg_latency_seconds
- model_violation_count
"""
from flask_restful import Resource
from flask import make_response, request, jsonify
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import func

from src import storage
from src.storage.models_labels import ModelLabel
from src.storage.outputs import Output
from src.storage.inference_periods import InferencePeriod
from src.storage.inputs import Input
from src.storage.ai_models import AIModel
from src.storage.labels import Label

# -------------------- Prometheus Gauges -------------------- #
MODEL_LABEL_GAUGE = Gauge(
    "label_model_metric",
    "Model metric value (accuracy|precision|recall|f1) as fraction",
    ["model_name", "label_name", "metric"]
)

AVG_CONF_GAUGE = Gauge(
    "label_model_avg_confidence",
    "Average prediction confidence for (model,label,approach)",
    ["model_name", "label_name", "approach"]
)

LATENCY_GAUGE = Gauge(
    "label_model_avg_latency_seconds",
    "Average inference latency (seconds) per model",
    ["model_name"]
)

VIOLATION_COUNT_GAUGE = Gauge(
    "model_violation_count",
    "Number of violations per model from input table",
    ["model_name"]
)

# -------------------- Helper Functions -------------------- #
def _resolve_names(sess, model_ids=None, label_ids=None):
    """Bulk resolve model and label names from IDs."""
    models_map, labels_map = {}, {}
    if model_ids:
        models_map = {m.id: m.name for m in sess.query(AIModel.id, AIModel.name)
                      .filter(AIModel.id.in_(model_ids)).all()}
    if label_ids:
        labels_map = {l.id: l.name for l in sess.query(Label.id, Label.name)
                      .filter(Label.id.in_(label_ids)).all()}
    return models_map, labels_map

# -------------------- Monitoring Classes -------------------- #
class Monitoring(Resource):
    """Prometheus scrape endpoint for AI model metrics."""

    def get(self):
        """
        Return Prometheus-formatted metrics.
        ---
        tags:
          - Monitoring
        summary: Metrics for Prometheus scraping
        responses:
          200:
            description: Prometheus metrics in text/plain
        """
        try:
            self.compute_metrics()
        except Exception as e:
            return make_response(f"# Error computing metrics: {str(e)}\n", 500,
                                 {"Content-Type": "text/plain"})
        return make_response(generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST})

    @staticmethod
    def compute_metrics():
        """Compute all metrics from DB and populate Prometheus gauges."""
        sess = storage.database.session

        # ----- 1. ModelLabel aggregated metrics -----
        ml_rows = sess.query(
            ModelLabel.ai_model_id,
            ModelLabel.label_id,
            func.avg(ModelLabel.accuracy).label("avg_accuracy"),
            func.avg(ModelLabel.precision).label("avg_precision"),
            func.avg(ModelLabel.recall).label("avg_recall"),
            func.avg(ModelLabel.f1_score).label("avg_f1")
        ).group_by(ModelLabel.ai_model_id, ModelLabel.label_id).all()

        model_ids = {r.ai_model_id for r in ml_rows}
        label_ids = {r.label_id for r in ml_rows}
        models_map, labels_map = _resolve_names(sess, model_ids, label_ids)

        for r in ml_rows:
            mname = models_map.get(r.ai_model_id, r.ai_model_id)
            lname = labels_map.get(r.label_id, r.label_id)
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="accuracy").set(r.avg_accuracy or 0.0)
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="precision").set(r.avg_precision or 0.0)
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="recall").set(r.avg_recall or 0.0)
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="f1_score").set(r.avg_f1 or 0.0)

        # ----- 2. Average confidence per model/label/approach -----
        conf_rows = sess.query(
            Output.ai_model_id,
            Output.label_id,
            Input.is_few_shot,
            Input.is_zero_shot,
            func.avg(Output.confidence).label("avg_conf")
        ).join(Input, Output.input_id == Input.id
        ).group_by(Output.ai_model_id, Output.label_id, Input.is_few_shot, Input.is_zero_shot).all()

        for r in conf_rows:
            mname = models_map.get(r.ai_model_id, r.ai_model_id)
            lname = labels_map.get(r.label_id, r.label_id)
            approach = "few_shot" if r.is_few_shot else "zero_shot" if r.is_zero_shot else "none"
            AVG_CONF_GAUGE.labels(model_name=mname, label_name=lname, approach=approach).set(r.avg_conf or 0.0)

        # ----- 3. Average latency per model -----
        lat_rows = sess.query(
            InferencePeriod.ai_model_id,
            func.avg(InferencePeriod.value).label("avg_latency")
        ).group_by(InferencePeriod.ai_model_id).all()

        for r in lat_rows:
            mname = models_map.get(r.ai_model_id, r.ai_model_id)
            LATENCY_GAUGE.labels(model_name=mname).set(r.avg_latency or 0.0)

        # ----- 4. Violation count per model -----
        violation_rows = sess.query(
            Output.ai_model_id,
            func.sum(Input.violation_count).label("total_violations")
        ).join(Input, Output.input_id == Input.id
        ).group_by(Output.ai_model_id).all()

        for r in violation_rows:
            mname = models_map.get(r.ai_model_id, r.ai_model_id)
            VIOLATION_COUNT_GAUGE.labels(model_name=mname).set(r.total_violations or 0)

# -------------------- JSON Summary Endpoint -------------------- #
class MetricsSummary(Resource):
    """Return JSON summary of metrics per model/label including violations."""

    def get(self):
        """
        JSON summary of metrics.
        ---
        tags:
          - Monitoring
        summary: Metrics aggregated per model and label
        parameters:
          - in: query
            name: top_n_labels
            schema:
              type: integer
            description: Return top N labels per model by average confidence (default 10)
        responses:
          200:
            description: Aggregated metrics JSON
        """
        sess = storage.database.session
        top_n = int(request.args.get("top_n_labels", 10))

        # ----- 1. Aggregate confidence per model/label -----
        conf_q = sess.query(
            Output.ai_model_id,
            Output.label_id,
            func.avg(Output.confidence).label("avg_conf"),
            func.count(Output.id).label("n")
        ).group_by(Output.ai_model_id, Output.label_id).all()

        model_ids = {r.ai_model_id for r in conf_q}
        label_ids = {r.label_id for r in conf_q}
        models_map, labels_map = _resolve_names(sess, model_ids, label_ids)

        # ----- 2. Aggregate violation counts per model -----
        violation_rows = sess.query(
            Output.ai_model_id,
            func.sum(Input.violation_count).label("total_violations")
        ).join(Input, Output.input_id == Input.id
        ).group_by(Output.ai_model_id).all()
        violation_map = {r.ai_model_id: r.total_violations for r in violation_rows}

        # ----- 3. Build summary per model -----
        results = {}
        for r in conf_q:
            mname = models_map.get(r.ai_model_id, r.ai_model_id)
            lname = labels_map.get(r.label_id, r.label_id)
            results.setdefault(mname, []).append({
                "label": lname,
                "avg_confidence": float(r.avg_conf),
                "count": int(r.n)
            })

        # ----- 4. Keep top N labels per model + violation count -----
        summary = {}
        for mname, labels in results.items():
            top_labels = sorted(labels, key=lambda x: x["avg_confidence"], reverse=True)[:top_n]
            summary[mname] = {
                "labels": top_labels,
                "violation_count": int(violation_map.get(model_ids.pop(), 0))
            }

        return jsonify(summary)
