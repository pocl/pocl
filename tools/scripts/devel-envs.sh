#!/bin/sh
libs_subdir=.

# source this while in the pocl build dir
export POCL_BUILDING=1
export OCL_ICD_VENDORS=$PWD/ocl-vendors

# AMDSDK supports the overriding via other env name.
export OPENCL_VENDOR_PATH=$OCL_ICD_VENDORS

# pocl test-cases don't link against pthreads, but libpocl does.
# this confuses gdb unless we preload libpthread.
# Not having this also makes cl2.hpp throw std::system_error exception for
# an unknown reason (at least on Ubuntu 14.04 / gcc 4.8.4).
# If libpocl is not built yet, this will fail...
export LD_PRELOAD=$(ldd lib/CL/$libs_subdir/libpocl.so | grep pthread | cut -f 3 -d' ')

#sometimes useful variable when ICD fails (and we use ocl-icd)
#export OCL_ICD_DEBUG=15
