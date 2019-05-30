#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"
VIRTUALENV="$4"
PYTH="$5"

cd "${TS_BASEDIR}/src" && "${VIRTUALENV}" --system-site-packages "--python=$PYTH" "PyOpenCL-build" && source "${TS_BUILDDIR}/bin/activate" && cd "${TS_SRCDIR}" && pip install pybind11 pytest && python3 configure.py
