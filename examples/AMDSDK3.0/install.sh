#!/bin/bash

BASEDIR=$1
BUILD_TYPE=$2

if [ ! -d "$BASEDIR" ]; then
  echo "USAGE: $0 AMD_SDK_BASEDIR"
  exit 1
fi

BINDIR="${BASEDIR}/AMD-APP-SDK-3.0/samples/opencl/bin/x86_64"

if [ -d "$BINDIR" ]; then
  echo "## Removing files from $BINDIR"
  rm -f $BINDIR/*
else
  mkdir -p $BINDIR
fi

FIND=$(which find)

if [ ! -x "$FIND" ]; then
  echo "'find' command not found, exiting"
  exit 3
fi

COPYDIRS=$($FIND "${BASEDIR}/AMD-APP-SDK-3.0/samples/opencl" -type d -name $BUILD_TYPE)

for DD in $COPYDIRS; do
  echo "## Copying files from $DD"
  cp $DD/* $BINDIR
done
