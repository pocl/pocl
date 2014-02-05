#!/bin/sh
# NOTE:
# 1) Install the official SPIR generator version of Clang/LLVM:
#    https://github.com/KhronosGroup/SPIR  
# 
# 2) Download opencl_spir.h from 
#    https://raw.github.com/KhronosGroup/SPIR-Tools/master/headers/opencl_spir.h
#    and add "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" in the beginning of
#    it to make it compile. 
clang -cc1 -emit-llvm-bc -triple spir-unknown-unknown -include opencl_spir.h -o example1.spir example1.cl
