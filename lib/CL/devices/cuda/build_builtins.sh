#!/bin/bash

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

nvcc --generate-code code=sm_52,arch=compute_52 --ptx --m64 --output-file $RELPATH/builtins.ptx $RELPATH/builtins.cu

nvcc --generate-code code=sm_70,arch=compute_70  --ptx --m64 --output-file $RELPATH/builtins_tensor.ptx $RELPATH/builtins.cu
