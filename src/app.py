#!/usr/bin/python3
"""Application intialisation module"""

from flask import Flask
from os import getenv
from flask import Flask, jsonify, make_response, Response
from src import storage
from flask_restful import Api
from flask_cors import CORS
from flasgger import Swagger
from .docs.swagger_template import swagger_template
from .config import config
import logging
from flask import send_from_directory
import os
from datetime import datetime
from .monitoring.metrics import get_metrics, get_metrics_content_type, refresh_quality_metrics_from_database, refresh_fewshot_metrics_from_database


# create the app instance
app = Flask(__name__)

# Configure Flask app
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['DEBUG'] = config.DEBUG
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

# Setup Swagger
swagger = Swagger(app, template=swagger_template)

# Setup CORS
cors = CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

@app.errorhandler(404)
def page_not_found(e):
    """json 404 page"""
    return make_response(jsonify({"error": "Resource (endpoint Not) found"}), 404)

@app.teardown_appcontext
def teardown(self) -> None:
    """Close the storage session"""
    storage.database.close()

@app.errorhandler(400)
def handle_bad_request(e):
    """json 400 page"""
    return (jsonify({'error': 'Bad request'}))

@app.route('/health')
def health_check():
    """
    Health check endpoint
    ---
    tags:
      - System
    summary: Check API health status
    description: Returns the current health status of the API and its components
    responses:
      200:
        description: API is healthy
        schema:
          $ref: '#/definitions/HealthResponse'
    """
    return jsonify({
        'status': 'healthy',
        'message': 'AI Object Counting API is running',
        'pipeline_available': True,
        'database': 'connected',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/metrics')
def metrics():
    """
    Prometheus metrics endpoint
    ---
    tags:
      - System
    summary: Get Prometheus-compatible metrics
    description: Returns OpenMetrics-compatible metrics for Prometheus monitoring
    responses:
      200:
        description: Metrics data in OpenMetrics format
        schema:
          type: string
          example: |
            # HELP api_requests_total Total number of API requests
            # TYPE api_requests_total counter
            api_requests_total 150
      500:
        description: Error generating metrics
        schema:
          type: string
          example: "# Error generating metrics"
    """
    try:
        # Refresh metrics from database before serving
        refresh_quality_metrics_from_database()
        refresh_fewshot_metrics_from_database()
        
        metrics_data = get_metrics()
        return Response(
            metrics_data,
            mimetype=get_metrics_content_type()
        )
    except Exception as e:
        logging.error(f"Failed to generate metrics: {e}")
        return Response(
            "# Error generating metrics\n",
            status=500,
            mimetype='text/plain'
        )

# setup the API and the endpoints
api = Api(app)
from .api.views.inputs import *
from .api.views.object_types import *
from .api.views.outputs import *
from .api.views.monitoring import PerformanceMetrics, ObjectTypeStats, DatabaseStats, ResetStats, SystemHealth
from .api.views.batch_processing import BatchProcessing, BatchStatus
from .api.views.fewshot import FewShotRegister, FewShotCount, FewShotObjectTypes, FewShotObjectTypeSingle, FewShotPredictions

api.add_resource(InputList, '/api/count')
# Add count-all endpoint for auto-detection
app.add_url_rule('/api/count-all', 'count_all_objects', InputList().count_all_objects, methods=['POST'])
# api.add_resource(InputSingle, '/api/input/<id>')
api.add_resource(ObjectTypeList, '/api/object-types')
api.add_resource(ObjectTypeSingle, '/api/object/<string:obj_id>')
api.add_resource(OutputList, '/api/results')
# Single result operations: register both URLs on one resource with a single endpoint name
api.add_resource(
    OutputSingle,
    '/api/results/<string:output_id>',
    '/api/correct/<string:output_id>',
    endpoint='output_single'
)

# Performance monitoring endpoints
api.add_resource(PerformanceMetrics, '/api/performance/metrics')
api.add_resource(ObjectTypeStats, '/api/performance/object-types')
api.add_resource(DatabaseStats, '/api/performance/database')
api.add_resource(ResetStats, '/api/performance/reset')
api.add_resource(SystemHealth, '/api/performance/health')

# Batch processing endpoints
api.add_resource(BatchProcessing, '/api/batch/process')
api.add_resource(BatchStatus, '/api/batch/status')

# Few-shot learning endpoints
api.add_resource(FewShotRegister, '/api/fewshot/register')
api.add_resource(FewShotCount, '/api/fewshot/count')
api.add_resource(FewShotObjectTypes, '/api/fewshot/object-types')
api.add_resource(FewShotObjectTypeSingle, '/api/fewshot/object-types/<string:object_name>')
api.add_resource(FewShotPredictions, '/api/fewshot/predictions')

# Serve media files
@app.route('/media/<path:filename>')
def serve_media(filename):
    """
    Serve media files
    ---
    tags:
      - System
    summary: Serve uploaded media files
    description: Serves uploaded image files from the media directory
    parameters:
      - in: path
        name: filename
        type: string
        required: true
        description: Path to the media file
    responses:
      200:
        description: Media file served successfully
        schema:
          type: file
      404:
        description: Media file not found
    """
    media_dir = config.MEDIA_DIRECTORY if os.path.isabs(config.MEDIA_DIRECTORY) else os.path.join(os.getcwd(), config.MEDIA_DIRECTORY)
    return send_from_directory(media_dir, filename)

# run this file to run the app
if __name__ == "__main__":
    logging.info(f"Starting AI Object Counting Application in {config.ENV} mode")
    logging.info(f"Server will run on {config.HOST}:{config.PORT}")
    logging.info(f"Debug mode: {config.DEBUG}")
    logging.info(f"Database: {config.DATABASE_TYPE}")
    logging.info(f"Media directory: {config.MEDIA_DIRECTORY}")
    
    app.run(
        host=config.HOST, 
        port=config.PORT, 
        threaded=True, 
        debug=config.DEBUG
    )
