#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"
VIRTUALENV="$4"
PYTH="$5"

cd "${TS_BASEDIR}/src" && "${VIRTUALENV}" "--python=$PYTH" "PyOpenCL-build"
