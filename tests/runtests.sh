#!/usr/bin/env bash
export COVERAGE_FILE=".coverage"
coverage run --source="matid" testrunner.py
unittest=$?
coverage run -m --source="matid" --append pytest
pytest=$?
if [ "$unittest" != 0 ] || [ "$pytest" != 0 ]; then
    exit 1
fi
coverage lcov -o ../coverage/lcov.info
