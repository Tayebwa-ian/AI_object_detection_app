#!/usr/bin/python3
"""API registration module.

Registers all Flask-RESTful resources with the Flask app in a clean, versioned way.
"""

from flask_restful import Api

# Inputs
from src.api.views.inputs import InputList, InputSingle

# Labels
from src.api.views.labels import LabelList, LabelSingle

# Outputs (predictions)
from src.api.views.outputs import OutputList, OutputSingle

# Monitoring endpoints
from src.api.views.monitoring import Monitoring, MetricsSummary

# AI Model CRUD
from src.api.views.model_registration import AIModelList, AIModelSingle

# Training endpoint
from src.api.views.models import MultiStageTrain


def register_api(app):
    """
    Register all API resources with the Flask application.

    Routes follow versioned API paths: /api/v1/...
    """
    api = Api(app)

    # Inputs
    api.add_resource(InputList, "/api/v1/inputs")
    api.add_resource(InputSingle, "/api/v1/inputs/<string:input_id>")

    # Labels
    api.add_resource(LabelList, "/api/v1/labels")
    api.add_resource(LabelSingle, "/api/v1/labels/<string:label_id>")

    # Outputs
    api.add_resource(OutputList, "/api/v1/outputs")
    api.add_resource(OutputSingle, "/api/v1/outputs/<string:output_id>")

    # Monitoring endpoints
    api.add_resource(Monitoring, "/metrics")
    api.add_resource(MetricsSummary, "/api/v1/metrics/summary")

    # AI Model CRUD
    api.add_resource(AIModelList, "/api/v1/models")
    api.add_resource(AIModelSingle, "/api/v1/models/<string:model_id>")

    # AI train
    api.add_resource(MultiStageTrain, "/api/v1/train")
