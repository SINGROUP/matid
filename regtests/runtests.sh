#!/usr/bin/env bash
coverage run -data-file=".coverage" --source="matid" testrunner.py
coverage lcov -data-file=".coverage" -o ../.coverage/lcov.info
