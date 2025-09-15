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

# Optional: run mutation tests if mutmut is installed
if command -v mutmut >/dev/null 2>&1; then
  echo "mutmut detected, running mutation tests (this can be slow)..."
  # Run mutmut in the project root. This will attempt to mutate code and run tests repeatedly.
  # You can tune mutmut options as needed (e.g. number of processes, tests per mutation).
  mutmut run
  echo "mutmut results: "
  mutmut results
else
  echo "mutmut not found -- skip mutation testing. To enable install 'mutmut' (pip install mutmut)."
fi

echo "All done."
