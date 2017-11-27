#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"

cd "${TS_SRCDIR}/test" && source "${TS_BUILDDIR}/bin/activate" && py.test -v --tb=native
