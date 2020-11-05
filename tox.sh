#!/bin/bash

sh clean.sh 
if (tox); then
    python -m pybadges \
        --left-text=build \
        --right-text=passing \
        --right-color=green \
        > docs/badges/build.svg
else
    python -m pybadges \
        --left-text=build \
        --right-text=failing \
        --right-color='#c00' \
        > docs/badges/build.svg
fi
