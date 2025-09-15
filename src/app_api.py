#!/usr/bin/python3
"""API registration module.

Creates the Flask-RESTful Api and adds resource routes. Keep this file small and
use it from `create_app()` so resources are registered cleanly.
"""
from flask_restful import Api

# Import resources (these modules must exist)
from src.api.views.inputs import InputList, InputSingle
from src.api.views.labels import LabelList, LabelSingle
from src.api.views.outputs import OutputList, OutputSingle
from src.api.views.monitoring import Monitoring, MetricsSummary, MetricsQuery
from src.api.views.models import *


def register_api(app):
    """Register all API resources with the Flask app.

    Routes used here conform to versioned API paths (/api/v1/...).
    """
    api = Api(app)

    # Inputs
    api.add_resource(InputList, "/api/v1/inputs")
    api.add_resource(InputSingle, "/api/v1/inputs/<string:input_id>")

    # Labels (object types)
    api.add_resource(LabelList, "/api/v1/labels")
    api.add_resource(LabelSingle, "/api/v1/labels/<string:label_id>")

    # Outputs (predictions)
    api.add_resource(OutputList, "/api/v1/outputs")
    api.add_resource(OutputSingle, "/api/v1/outputs/<string:output_id>")

    # Monitoring: Prometheus scrape endpoint + JSON summaries + query endpoint
    api.add_resource(Monitoring, "/metrics")
    api.add_resource(MetricsSummary, "/api/v1/metrics/summary")
    api.add_resource(MetricsQuery, "/api/v1/metrics/query")

        # models
    api.add_resource(AIModelList, "/api/v1/models")
    api.add_resource(AIModelSingle, "/api/v1/models/<string:model_id>")
    api.add_resource(TrainModel, "/api/v1/models/<string:model_id>/train")
    api.add_resource(TestModel, "/api/v1/models/<string:model_id>/test")

