#!/bin/bash

TS_BASEDIR="$1"
shift
TS_BUILDDIR="$1"
shift
TS_SRCDIR="$1"
shift

# Set DBG_CMD="gdb --args" to drop to a debugger.
# -s is useful as it dumps the PoCL logs directly without capture.
# -k can be used to filter the test

cd "${TS_SRCDIR}/test" && ${DBG_CMD} ${TS_BUILDDIR}/bin/python3 ${TS_BUILDDIR}/bin/py.test -s -v --tb=native $@
