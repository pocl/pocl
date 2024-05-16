#!/bin/sh
libs_subdir=.

# source this while in the pocl build dir
export POCL_BUILDING=1
# Old OCL ICD Loaders may not recognize PoCL without the trailing '/'.
export OCL_ICD_VENDORS=$PWD/ocl-vendors/

# AMDSDK supports the overriding via other env name.
export OPENCL_VENDOR_PATH=$OCL_ICD_VENDORS

#sometimes useful variable when ICD fails (and we use ocl-icd)
#export OCL_ICD_DEBUG=15
export PATH=$PWD/bin:$PATH
