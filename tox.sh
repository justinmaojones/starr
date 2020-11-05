#!/bin/bash

sh clean.sh 
BUILD_STATUS=docs/badges/build.svg
if (tox); then
    python -m pybadges \
        --left-text=build \
        --right-text=passing \
        --right-color=green \
        > $BUILD_STATUS
else
    python -m pybadges \
        --left-text=build \
        --right-text=failing \
        --right-color='#c00' \
        > $BUILD_STATUS
fi
git add $BUILD_STATUS
