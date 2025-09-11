#!/usr/bin/python3
"""Application intialisation module"""

from flask import Flask
from os import getenv
from flask import Flask, jsonify, make_response
from src import storage
from flask_restful import Api
from flask_cors import CORS
from flasgger import Swagger
from .docs.swagger_template import swagger_template


# create the app instance
app = Flask(__name__)
swagger = Swagger(app, template= swagger_template)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

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

# setup the API and the endpoints
api = Api(app)
from .api.views.inputs import *
from .api.views.object_types import *
from .api.views.outputs import *

api.add_resource(InputList, '/api/count')
# api.add_resource(InputSingle, '/api/input/<id>')
api.add_resource(ObjectTypeList, '/api/object')
api.add_resource(ObjectTypeSingle, '/api/object/<id>')
api.add_resource(OutputList, '/api/output')
api.add_resource(OutputSingle, '/api/correct/<id>')
#api.add_resource(Monitoring, '/metrics')

# run this file to run the app
if __name__ == "__main__":
    host = getenv("OBJ_DETECT_API_HOST", "0.0.0.0")
    port = int(getenv("OBJ_DETECT_API_PORT", "5000"))
    app.run(host, port=port, threaded=True, debug=True)
