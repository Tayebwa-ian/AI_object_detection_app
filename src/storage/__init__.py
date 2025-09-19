#!/usr/bin/python3
"""Storage package initializer.

Creates a single `database` Engine instance for the application.
The module exports:
 - database: an Engine instance (call database.reload() if you changed DATABASE_URL at runtime)
 - BaseModel and model classes for convenience (optional)
"""
from os import getenv
from .engine.engine import Engine
from .base_model import BaseModel, Base

# instantiate Engine
database = Engine()

# If user expects session immediately, create session (safe to call multiple times)
database.reload()

# Expose common models for convenience (optional; can also import from module files)
from .ai_models import AIModel
from .inputs import Input
from .labels import Label
from .outputs import Output
from .models_labels import ModelLabel
from .inference_periods import InferencePeriod
from .evaluation_runs import EvaluationRun
