#!/usr/bin/env bash
# run_tests.sh - run unittest test suite with coverage and optional mutation testing
# Usage: ./run_tests.sh

set -euo pipefail

# Ensure dev/test env
export OBJ_DETECT_ENV=test
export OBJ_DETECT_USE_SQLITE=1
# In-memory sqlite is chosen by Engine when OBJ_DETECT_ENV=test
export OBJ_DETECT_SQLITE_FILE=":memory:"

echo "Running unit tests with coverage (branch)..."

# Run tests with coverage - branch coverage enabled
coverage run --branch -m unittest discover -s tests -v

echo ""
echo "Coverage report (terminal):"
coverage report -m

# generate HTML report
coverage html -d coverage_html
echo "HTML coverage written to coverage_html/index.html"

echo "All done."
