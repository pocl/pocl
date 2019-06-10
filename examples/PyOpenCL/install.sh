#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"

cd "${TS_SRCDIR}" && ${TS_BUILDDIR}/bin/python3 setup.py install
