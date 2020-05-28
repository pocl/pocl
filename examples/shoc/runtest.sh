#!/bin/bash

TS_BASEDIR="$1"
TS_BUILDDIR="$2"
TS_SRCDIR="$3"
TEST_NAME="$4"

cd "${TS_SRCDIR}" && ./bin/shocdriver -opencl -benchmark ${TEST_NAME} -s 4
