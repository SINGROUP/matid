#!/usr/bin/env bash
cd regtests
export COVERAGE_FILE="../.coverage"
coverage run --source="matid" testrunner.py
coverage lcov
