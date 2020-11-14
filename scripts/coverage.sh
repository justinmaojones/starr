#!/bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately

# run from root
coverage run -m pytest
coverage report -m
coverage-badge -f -o ./docs/badges/coverage.svg
