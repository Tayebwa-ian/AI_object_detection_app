#!/usr/bin/env bash
# dev_run.sh - Development runner for the Object Counting API
# Usage: ./dev_run.sh

set -euo pipefail

# Development environment variables (override as needed)
export OBJ_DETECT_USE_SQLITE=1
export OBJ_DETECT_SQLITE_FILE="./Databases/obj_detect_dev.db"
export OBJ_DETECT_ENV=development
export OBJ_DETECT_API_HOST="127.0.0.1"
export OBJ_DETECT_API_PORT=5000
export OBJ_DETECT_DEBUG=1

# Create data directory if missing
mkdir -p ./Databases
echo "Starting development server with SQLite DB at ${OBJ_DETECT_SQLITE_FILE}..."
python -m src.app
