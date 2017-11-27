#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"

source "${TS_BUILDDIR}/bin/activate" &&  cd "${TS_SRCDIR}" &&  python setup.py install
