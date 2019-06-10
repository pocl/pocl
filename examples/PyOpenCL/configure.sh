#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"
VIRTUALENV="$4"
PYTH="$5"

cd "${TS_BASEDIR}/src" && "${VIRTUALENV}" --no-site-packages "--python=$PYTH" "PyOpenCL-build" && cd "${TS_SRCDIR}" && ${TS_BUILDDIR}/bin/pip3 install mako pybind11 pytest && ${TS_BUILDDIR}/bin/python3 configure.py
