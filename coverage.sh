#!/bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately

coverage run -m pytest
coverage report -m
coverage-badge -f -o ./docs/badges/coverage.svg
