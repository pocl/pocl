#!/bin/bash

TS_BASEDIR="$1"
shift
TS_BUILDDIR="$1"
shift
TS_SRCDIR="$1"
shift

cd "${TS_SRCDIR}/test" && ${TS_BUILDDIR}/bin/py.test -v --tb=native $@
