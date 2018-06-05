#!/usr/bin/env zsh
workon matid
coverage run --source="../matid" regtests.py
