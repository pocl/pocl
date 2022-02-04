#!/bin/bash

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

# nvcc --generate-code code=sm_35,arch=compute_35 --ptx --m64 --output-file $RELPATH/builtins_sm35.ptx $RELPATH/builtins.cu

nvcc --generate-code code=sm_50,arch=compute_50 --ptx --m64 --output-file $RELPATH/builtins_sm50.ptx $RELPATH/builtins.cu

# nvcc --generate-code code=sm_60,arch=compute_60  --ptx --m64 --output-file $RELPATH/builtins_sm60.ptx $RELPATH/builtins.cu

nvcc --generate-code code=sm_70,arch=compute_70  --ptx --m64 --output-file $RELPATH/builtins_sm70.ptx $RELPATH/builtins.cu

# nvcc --generate-code code=sm_80,arch=compute_80  --ptx --m64 --output-file $RELPATH/builtins_sm80.ptx $RELPATH/builtins.cu
