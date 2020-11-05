coverage run -m pytest
coverage report -m
coverage-badge -f -o ./docs/badges/coverage.svg
git add ./docs/badges/coverage.svg
