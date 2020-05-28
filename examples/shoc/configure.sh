#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"
VIRTUALENV="$4"
PYTH="$5"

cd "${TS_SRCDIR}" && ./configure --with-opencl --without-cuda --without-mpi
