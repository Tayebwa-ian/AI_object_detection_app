#!/usr/bin/python3
"""Monitoring endpoints for metrics and aggregations.

Provides:
- /metrics                -> Prometheus scrape (text format) (Flask-RESTful Resource: Monitoring)
- /api/v1/metrics/summary -> JSON summary of metrics (MetricsSummary)
- /api/v1/metrics/query   -> Flexible metrics query with filtering/grouping (MetricsQuery)

Design notes:
- Aggregations are performed at the DB level (SQL GROUP BY) to remain efficient for large datasets.
- Name resolution (id -> human name) is done in bulk to avoid N+1 queries.
- Prometheus Gauges are created once at module import time to avoid re-creating metric families on each scrape.
"""
from flask_restful import Resource
from flask import make_response, request, jsonify
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import func, and_
from sqlalchemy.orm import load_only

from src import storage
from src.storage.models_labels import ModelLabel
from src.storage.outputs import Output
from src.storage.ai_models import AIModel
from src.storage.labels import Label
from src.storage.inference_periods import InferencePeriod
from src.storage.inputs import Input

# Prometheus metrics (defined once)
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


def _resolve_names(sess, model_ids=None, label_ids=None):
    """Resolve ai_model and label names in bulk.

    Args:
        sess: SQLAlchemy session
        model_ids: iterable of model ids to resolve
        label_ids: iterable of label ids to resolve

    Returns:
        (models_map, labels_map) where each is dict id -> name
    """
    models_map = {}
    labels_map = {}

    if model_ids:
        q = sess.query(AIModel.id, AIModel.name).filter(AIModel.id.in_(list(model_ids)))
        models_map = {m.id: m.name for m in q.all()}

    if label_ids:
        q = sess.query(Label.id, Label.name).filter(Label.id.in_(list(label_ids)))
        labels_map = {l.id: l.name for l in q.all()}

    return models_map, labels_map


class Monitoring(Resource):
    """Prometheus scrape endpoint (text format)."""

    def get(self):
        """
        Prometheus metrics endpoint.
        ---
        tags:
          - Monitoring
        summary: Prometheus scrape endpoint (text)
        description: Compute current aggregated metrics and return Prometheus-formatted metrics.
        responses:
          200:
            description: Prometheus metrics in text format (suitable for scrape)
        """
        # Compute and populate Gauges
        try:
            self.compute_metrics()
        except Exception as e:
            # If computation fails, return a 500 with a small message (still return prior metric contents if any)
            # Do not raise an unhandled exception because Prometheus expects HTTP 200/5xx responses.
            return make_response(f"# Error computing metrics: {str(e)}\n", 500, {"Content-Type": "text/plain"})

        # Return the metrics in the official Prometheus text format
        return make_response(generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST})

    @staticmethod
    def compute_metrics():
        """Compute metrics from DB (ModelLabel, Output, InferencePeriod) and set Prometheus gauges.

        - ModelLabel: averaged accuracy/precision/recall/f1 per (model,label)
        - Output: average confidence per (model,label,approach)
        - InferencePeriod: average latency per model
        """
        sess = storage.database.session

        # 1) ModelLabel aggregates (db-level average per ai_model_id,label_id)
        ml_rows = (
            sess.query(
                ModelLabel.ai_model_id,
                ModelLabel.label_id,
                func.avg(ModelLabel.accuracy).label("avg_accuracy"),
                func.avg(ModelLabel.precision).label("avg_precision"),
                func.avg(ModelLabel.recall).label("avg_recall"),
                func.avg(ModelLabel.f1_score).label("avg_f1")
            )
            .group_by(ModelLabel.ai_model_id, ModelLabel.label_id)
            .all()
        )

        model_ids = {r.ai_model_id for r in ml_rows}
        label_ids = {r.label_id for r in ml_rows}
        models_map, labels_map = _resolve_names(sess, model_ids=model_ids, label_ids=label_ids)

        # Populate MODEL_LABEL_GAUGE
        for r in ml_rows:
            mname = models_map.get(r.ai_model_id, r.ai_model_id)
            lname = labels_map.get(r.label_id, r.label_id)
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="accuracy").set(float(r.avg_accuracy or 0.0))
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="precision").set(float(r.avg_precision or 0.0))
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="recall").set(float(r.avg_recall or 0.0))
            MODEL_LABEL_GAUGE.labels(model_name=mname, label_name=lname, metric="f1_score").set(float(r.avg_f1 or 0.0))

        # 2) Average confidence per (model,label,approach)
        # Join Output -> Input to get approach flags (is_few_shot / is_zero_shot)
        conf_rows = (
            sess.query(
                Output.ai_model_id,
                Output.label_id,
                Input.is_few_shot,
                Input.is_zero_shot,
                func.avg(Output.confidence).label("avg_conf"),
                func.count(Output.id).label("n")
            )
            .join(Input, Output.input_id == Input.id)
            .group_by(Output.ai_model_id, Output.label_id, Input.is_few_shot, Input.is_zero_shot)
            .all()
        )

        conf_model_ids = {r.ai_model_id for r in conf_rows}
        conf_label_ids = {r.label_id for r in conf_rows}
        conf_models_map, conf_labels_map = _resolve_names(sess, model_ids=conf_model_ids, label_ids=conf_label_ids)

        for r in conf_rows:
            mname = conf_models_map.get(r.ai_model_id, r.ai_model_id)
            lname = conf_labels_map.get(r.label_id, r.label_id)
            # approach label
            if r.is_few_shot:
                approach = "few_shot"
            elif r.is_zero_shot:
                approach = "zero_shot"
            else:
                approach = "none"
            AVG_CONF_GAUGE.labels(model_name=mname, label_name=lname, approach=approach).set(float(r.avg_conf or 0.0))

        # 3) Average inference latency per model from InferencePeriod
        lat_rows = (
            sess.query(
                InferencePeriod.ai_model_id,
                func.avg(InferencePeriod.value).label("avg_latency"),
                func.count(InferencePeriod.id).label("n")
            )
            .group_by(InferencePeriod.ai_model_id)
            .all()
        )

        lat_model_ids = {r.ai_model_id for r in lat_rows}
        lat_models_map, _ = _resolve_names(sess, model_ids=lat_model_ids, label_ids=None)

        for r in lat_rows:
            mname = lat_models_map.get(r.ai_model_id, r.ai_model_id)
            LATENCY_GAUGE.labels(model_name=mname).set(float(r.avg_latency or 0.0))


class MetricsSummary(Resource):
    """Return a JSON summary of metrics useful for dashboards.

    Example output:
    {
        "model_name_1": [
            {"label": "car", "avg_confidence": 0.95, "count": 200},
            ...
        ],
        ...
    }
    """

    def get(self):
        """
        Get JSON metrics summary.
        ---
        tags:
          - Monitoring
        summary: JSON metrics summary
        parameters:
          - in: query
            name: top_n_labels
            schema:
              type: integer
            description: Return top N labels by average confidence per model (optional, default 10)
        responses:
          200:
            description: JSON aggregated metrics summary
        """
        sess = storage.database.session
        top_n = int(request.args.get("top_n_labels", 10))

        # Aggregate average confidence by (ai_model_id, label_id)
        conf_q = (
            sess.query(
                Output.ai_model_id.label("ai_model_id"),
                Output.label_id.label("label_id"),
                func.avg(Output.confidence).label("avg_conf"),
                func.count(Output.id).label("n")
            )
            .group_by(Output.ai_model_id, Output.label_id)
        )

        # Collect ids for bulk name resolution
        model_ids = set()
        label_ids = set()
        for row in conf_q:
            model_ids.add(row.ai_model_id)
            label_ids.add(row.label_id)

        models_map, labels_map = _resolve_names(sess, model_ids=model_ids, label_ids=label_ids)

        # Build results per model
        results = {}
        for row in conf_q:
            mname = models_map.get(row.ai_model_id, row.ai_model_id)
            lname = labels_map.get(row.label_id, row.label_id)
            results.setdefault(mname, []).append({
                "label": lname,
                "avg_confidence": float(row.avg_conf),
                "count": int(row.n)
            })

        # Sort lists by avg_confidence descending and keep top_n
        for m in results:
            results[m] = sorted(results[m], key=lambda x: x["avg_confidence"], reverse=True)[:top_n]

        return jsonify(results)


class MetricsQuery(Resource):
    """Flexible metrics query endpoint.

    Query parameters:
      - approach: few_shot | zero_shot | none
      - ai_model_id
      - label_id
      - input_id
      - metric: confidence | latency | accuracy | precision | recall | f1
      - agg: avg | min | max | count
      - group_by: comma separated values from model,label,input
    """

    VALID_METRICS = {"confidence", "latency", "accuracy", "precision", "recall", "f1"}
    VALID_AGGS = {"avg", "min", "max", "count"}
    VALID_GROUP_BY_PARTS = {"model", "label", "input"}

    def _parse_group_by(self, group_by_raw):
        """Return list of group_by parts in normalized order and validate them."""
        if not group_by_raw:
            return ["model", "label"]
        parts = [p.strip() for p in group_by_raw.split(",") if p.strip()]
        # Validate each
        for p in parts:
            if p not in self.VALID_GROUP_BY_PARTS:
                raise ValueError(f"invalid group_by part: {p}")
        return parts

    def get(self):
        """
        Query metrics with filters and grouping.
        ---
        tags:
          - Monitoring
        summary: Query aggregated metrics
        parameters:
          - in: query
            name: approach
            schema: { type: string }
            description: few_shot | zero_shot | none
          - in: query
            name: ai_model_id
            schema: { type: string }
          - in: query
            name: label_id
            schema: { type: string }
          - in: query
            name: input_id
            schema: { type: string }
          - in: query
            name: metric
            schema: { type: string, default: confidence }
            description: confidence | latency | accuracy | precision | recall | f1
          - in: query
            name: agg
            schema: { type: string, default: avg }
            description: avg|min|max|count
          - in: query
            name: group_by
            schema: { type: string, default: model,label }
            description: model | label | input | combinations comma-separated
        responses:
          200:
            description: Aggregated metrics in JSON
          400:
            description: Invalid parameters
        """
        sess = storage.database.session

        # Read & normalize params
        approach = (request.args.get("approach") or "").strip().lower() or None
        ai_model_id = request.args.get("ai_model_id")
        label_id = request.args.get("label_id")
        input_id = request.args.get("input_id")
        metric = (request.args.get("metric") or "confidence").strip().lower()
        agg = (request.args.get("agg") or "avg").strip().lower()
        group_by_raw = request.args.get("group_by", "model,label")

        # Validate
        if metric not in self.VALID_METRICS:
            return {"error": f"invalid metric. valid: {sorted(self.VALID_METRICS)}"}, 400
        if agg not in self.VALID_AGGS:
            return {"error": f"invalid agg. valid: {sorted(self.VALID_AGGS)}"}, 400

        try:
            group_by_parts = self._parse_group_by(group_by_raw)
        except ValueError as e:
            return {"error": str(e)}, 400

        # Build approach filter expression (to apply when joining Outputs -> Input)
        approach_filter = None
        if approach:
            if approach == "few_shot":
                approach_filter = Input.is_few_shot.is_(True)
            elif approach == "zero_shot":
                approach_filter = Input.is_zero_shot.is_(True)
            elif approach == "none":
                approach_filter = and_(Input.is_few_shot.is_(False), Input.is_zero_shot.is_(False))
            else:
                return {"error": "invalid approach, use few_shot | zero_shot | none"}, 400

        # Map agg to SQL function
        agg_map = {
            "avg": func.avg,
            "min": func.min,
            "max": func.max,
            "count": func.count
        }
        agg_func = agg_map[agg]

        # Helper to resolve name maps for ids appearing in rows
        def _resolve_maps_from_rows(rows, model_idx=None, label_idx=None):
            model_ids = set()
            label_ids = set()
            for r in rows:
                if model_idx is not None and r[model_idx] is not None:
                    model_ids.add(r[model_idx])
                if label_idx is not None and r[label_idx] is not None:
                    label_ids.add(r[label_idx])
            return _resolve_names(sess, model_ids=model_ids, label_ids=label_ids)

        # Build and execute different queries depending on metric
        # Prepare return format: list of dicts where keys match group_by parts plus "value" and "count"
        results = []

        if metric == "confidence":
            # Grouping columns are chosen based on requested group_by_parts
            select_cols = []
            group_cols = []
            col_map = {}  # mapping from semantic name to actual column object

            if "model" in group_by_parts:
                select_cols.append(Output.ai_model_id)
                group_cols.append(Output.ai_model_id)
                col_map["model"] = "ai_model_id"
            if "label" in group_by_parts:
                select_cols.append(Output.label_id)
                group_cols.append(Output.label_id)
                col_map["label"] = "label_id"
            if "input" in group_by_parts:
                select_cols.append(Output.input_id)
                group_cols.append(Output.input_id)
                col_map["input"] = "input_id"

            # Always include aggregated value and count as trailing columns
            select_cols.append(agg_func(Output.confidence).label("value"))
            select_cols.append(func.count(Output.id).label("n"))

            q = sess.query(*select_cols)
            # Join Input if approach_filter present (to access approach flags)
            if approach_filter is not None:
                q = q.join(Input, Output.input_id == Input.id).filter(approach_filter)

            # Apply simple filters
            if ai_model_id:
                q = q.filter(Output.ai_model_id == ai_model_id)
            if label_id:
                q = q.filter(Output.label_id == label_id)
            if input_id:
                q = q.filter(Output.input_id == input_id)

            # Apply grouping
            if group_cols:
                q = q.group_by(*group_cols)

            rows = q.all()

            # Build name maps for resolution
            # Determine model_idx and label_idx positions in returned row tuple
            model_idx = None
            label_idx = None
            # positions correspond to order in select_cols
            pos = 0
            for part in group_by_parts:
                if part == "model":
                    model_idx = pos
                if part == "label":
                    label_idx = pos
                pos += 1
            # value at pos, n at pos+1 (depending on how many group columns)
            models_map, labels_map = _resolve_maps_from_rows(rows, model_idx=model_idx, label_idx=label_idx)

            for r in rows:
                # tuple: (g1, g2, ..., value, n)
                # map group parts to keys
                entry = {}
                idx = 0
                for part in group_by_parts:
                    val = r[idx]
                    if part == "model":
                        entry["ai_model_id"] = val
                        entry["ai_model_name"] = models_map.get(val, None) if val is not None else None
                    elif part == "label":
                        entry["label_id"] = val
                        entry["label_name"] = labels_map.get(val, None) if val is not None else None
                    elif part == "input":
                        entry["input_id"] = val
                    idx += 1
                # aggregated columns
                value = r[idx]
                count = r[idx + 1]
                entry["value"] = float(value) if value is not None else None
                entry["count"] = int(count)
                results.append(entry)

            return jsonify({"metric": "confidence", "agg": agg, "group_by": group_by_parts, "results": results})

        elif metric == "latency":
            # For latency we group by requested parts but mainly supported grouping is by model and input.
            select_cols = []
            group_cols = []

            if "model" in group_by_parts:
                select_cols.append(InferencePeriod.ai_model_id)
                group_cols.append(InferencePeriod.ai_model_id)
            if "input" in group_by_parts:
                select_cols.append(InferencePeriod.input_id)
                group_cols.append(InferencePeriod.input_id)
            if "label" in group_by_parts:
                # latency isn't directly associated with label table; ignore label grouping if requested
                # but we keep a consistent behavior: return 400 if label grouping requested for latency
                return {"error": "grouping by 'label' is not supported for 'latency' metric"}, 400

            select_cols.append(agg_func(InferencePeriod.value).label("value"))
            select_cols.append(func.count(InferencePeriod.id).label("n"))

            q = sess.query(*select_cols)
            if ai_model_id:
                q = q.filter(InferencePeriod.ai_model_id == ai_model_id)
            if input_id:
                q = q.filter(InferencePeriod.input_id == input_id)

            if group_cols:
                q = q.group_by(*group_cols)

            rows = q.all()

            # bulk resolve model names if present
            model_ids = {r[0] for r in rows} if rows and "model" in group_by_parts else set()
            models_map, _ = _resolve_names(sess, model_ids=model_ids, label_ids=None)

            for r in rows:
                entry = {}
                idx = 0
                if "model" in group_by_parts:
                    entry["ai_model_id"] = r[idx]
                    entry["ai_model_name"] = models_map.get(r[idx], None)
                    idx += 1
                if "input" in group_by_parts:
                    entry["input_id"] = r[idx]
                    idx += 1
                entry["value"] = float(r[idx]) if r[idx] is not None else None
                entry["count"] = int(r[idx + 1])
                results.append(entry)

            return jsonify({"metric": "latency", "agg": agg, "group_by": group_by_parts, "results": results})

        else:
            # accuracy|precision|recall|f1: ModelLabel table
            metric_col_map = {
                "accuracy": ModelLabel.accuracy,
                "precision": ModelLabel.precision,
                "recall": ModelLabel.recall,
                "f1": ModelLabel.f1_score
            }
            col = metric_col_map.get(metric)
            if col is None:
                return {"error": "unsupported metric"}, 400

            select_cols = []
            group_cols = []
            if "model" in group_by_parts:
                select_cols.append(ModelLabel.ai_model_id)
                group_cols.append(ModelLabel.ai_model_id)
            if "label" in group_by_parts:
                select_cols.append(ModelLabel.label_id)
                group_cols.append(ModelLabel.label_id)
            if "input" in group_by_parts:
                # model_label table is not per-input; grouping by input is not supported
                return {"error": "grouping by 'input' is not supported for model-level metrics"}, 400

            select_cols.append(agg_func(col).label("value"))
            select_cols.append(func.count(ModelLabel.id).label("n"))

            q = sess.query(*select_cols)
            if ai_model_id:
                q = q.filter(ModelLabel.ai_model_id == ai_model_id)
            if label_id:
                q = q.filter(ModelLabel.label_id == label_id)

            if group_cols:
                q = q.group_by(*group_cols)

            rows = q.all()
            models_map, labels_map = _resolve_maps_from_rows(rows, model_idx=0 if "model" in group_by_parts else None, label_idx=0 if ("label" in group_by_parts and "model" not in group_by_parts) else (1 if ("label" in group_by_parts and "model" in group_by_parts) else None))

            for r in rows:
                entry = {}
                idx = 0
                if "model" in group_by_parts:
                    entry["ai_model_id"] = r[idx]
                    entry["ai_model_name"] = models_map.get(r[idx], None)
                    idx += 1
                if "label" in group_by_parts:
                    entry["label_id"] = r[idx]
                    entry["label_name"] = labels_map.get(r[idx], None)
                    idx += 1
                entry["value"] = float(r[idx]) if r[idx] is not None else None
                entry["count"] = int(r[idx + 1])
                results.append(entry)

            return jsonify({"metric": metric, "agg": agg, "group_by": group_by_parts, "results": results})
