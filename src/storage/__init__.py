#!/usr/bin/python3
"""create a unique Storage instance for the application"""
from .engine import engine
from .base_model import BaseModel
from .inputs import Input
from .object_types import ObjectType
from .ouputs import Output
from os import getenv


# load from database
db = getenv('OBJ_DETECT_MYSQL_DB')
if db:
    database = engine.Engine()
    database.reload()
