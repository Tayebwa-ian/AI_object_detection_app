#!/usr/bin/python3
"""create a unique Storage instance for the application"""
from .engine import engine
from .base_model import BaseModel
from .inputs import Input
from .object_types import ObjectType
from .outputs import Output
from .fewshot_models import FewShotObjectType, FewShotSupportImage, FewShotPrediction, FewShotLearningSession
from os import getenv, environ


# Load database configuration
db = getenv('OBJ_DETECT_MYSQL_DB')

# Provide a sensible default for development if not set
if not db:
    environ['OBJ_DETECT_MYSQL_DB'] = 'dev_obj_detect.db'

# Initialize database engine
database = engine.Engine()
database.reload()
