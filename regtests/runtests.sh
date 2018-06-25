#!/usr/bin/env bash
export COVERAGE_FILE="../.coverage"
coverage run --source="../matid" testrunner.py
