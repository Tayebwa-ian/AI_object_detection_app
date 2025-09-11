"""Monitoring Metrics"""
from flask_restful import Resource
from flask import make_response
from ...storage import *
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# Define metrics ONCE, with labels
accuracy_gauge = Gauge(
    'object_model_accuracy',
    'Model accuracy as a percentage',
    ['object_type']
)
precision_gauge = Gauge(
    'object_model_precision',
    'Model precision as a percentage',
    ['object_type']
)
recall_gauge = Gauge(
    'object_model_recall',
    'Model recall as a percentage',
    ['object_type']
)
avg_confidence_gauge = Gauge(
    'object_model_avg_pred_confidence',
    'Average prediction confidence',
    ['object_type']
)
sam_time_gauge = Gauge(
    'sam_inference_time_seconds',
    'SAM model inference time in seconds',
    ['object_type']
)
resnet_time_gauge = Gauge(
    'resnet_inference_time_seconds',
    'ResNet model inference time in seconds',
    ['object_type']
)
bart_time_gauge = Gauge(
    'bart_inference_time_seconds',
    'BART model inference time in seconds',
    ['object_type']
)

class Monitoring(Resource):
    """
    Setup metrics to monitor in this app using Prometheus
    """
    def get(self):
        """returns all setup metrics"""
        # Before exporting, recompute values
        Monitoring.compute_details()
        return make_response(
            generate_latest(),
            200,
            {'Content-Type': CONTENT_TYPE_LATEST}
        )

    @staticmethod
    def compute_details() -> None:
        """
        Calculates metrics per object type from DB
        """
        object_types = database.all(ObjectType)

        for obj in object_types:
            # Each ObjectType has related Metric rows
            metrics = obj.metrics  # via relationship
            print(metrics)
            if not metrics:
                print("Hi no metrics")
                continue

            # Example: average across all rows for this object type
            acc = sum(m.acurracy for m in metrics) / len(metrics)
            prec = sum(m.precision for m in metrics) / len(metrics)
            rec = sum(m.recall for m in metrics) / len(metrics)
            conf = sum(o.pred_confidence for o in obj.outputs) / len(obj.outputs)

            sam_time = sum(m.sam_inference_time for m in metrics) / len(metrics)
            resnet_time = sum(m.resnet_inference_time for m in metrics) / len(metrics)
            bart_time = sum(m.bart_inference_time or 0 for m in metrics) / len(metrics)

            # Update metrics with labels
            accuracy_gauge.labels(object_type=obj.name).set(acc)
            precision_gauge.labels(object_type=obj.name).set(prec)
            recall_gauge.labels(object_type=obj.name).set(rec)
            avg_confidence_gauge.labels(object_type=obj.name).set(conf)
            sam_time_gauge.labels(object_type=obj.name).set(sam_time)
            resnet_time_gauge.labels(object_type=obj.name).set(resnet_time)
            bart_time_gauge.labels(object_type=obj.name).set(bart_time)
