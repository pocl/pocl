#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"

source "${TS_BUILDDIR}/bin/activate" 
echo $PYTHONPATH
which python3
cd "${TS_SRCDIR}/test" && python3 /usr/bin/py.test-3 -v --tb=native
