#!/bin/bash

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

nvcc --ptx --m64 --output-file $RELPATH/builtins.ptx $RELPATH/builtins.cu
