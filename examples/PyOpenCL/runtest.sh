#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"

cd "${TS_SRCDIR}/test" && ${TS_BUILDDIR}/bin/py.test -v --tb=native
