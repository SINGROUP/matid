#!/usr/bin/env zsh
workon systax
coverage run --source="../systax" regtests.py
