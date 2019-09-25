Supported OpenCL features
=========================

All mandatory features for OpenCL 1.2 are supported
on x86-64+Linux, see :ref:`pocl-conformance` for details.

Known unsupported OpenCL features
=================================

The known unsupported OpenCL (both 1.x and 2.x) features are
listed here as encountered.

Frontend/Clang
--------------

* OpenCL 1.x

  * OpenGL interoperability extension
  * SPIR extension (partially available, see below)

* OpenCL 2.0

  * generic address space (recognized by LLVM 3.8+ but incomplete)
  * pipes (WIP)
  * device-side enqueue

* cl_khr_f16: half precision support (with the exception of  vload_half / vstore_half)

Unimplemented host side functions
---------------------------------

All 1.2 API call are implemented. From the 2.x API, only a small part
is implemented: clCreateProgramWithIL, clSetKernelArgSVMPointer, clSVM*
and clEnqueueSVM* calls.

SPIR support
------------

There is some experimental support available for SPIR. In particular,
for the CPU device, most non-vectorized variants of the kernel library
functions should work with SPIR bitcode; the rest is not implemented yet.
SPIR on TCE,CUDA and other devices is currently untested.

There is some even more experimental support available for loading SPIR-V
binaries, but this depends on functional llvm-spirv translator.
The translator can be downloaded & built here:

https://github.com/KhronosGroup/SPIRV-LLVM-Translator

Note that you must checkout the same branch as the LLVM which your pocl
is built against, otherwise pocl might refuse to load a binary.

Compiling from OpenCL sources to SPIR::

     clang -D_CL_DISABLE_HALF -Xclang -cl-std=CL1.2 -D__OPENCL_C_VERSION__=120
     -D__OPENCL_VERSION__=120 -Dcl_khr_int64 -Dcl_khr_byte_addressable_store
     -Dcl_khr_global_int32_base_atomics -Dcl_khr_global_int32_extended_atomics
     -Dcl_khr_local_int32_base_atomics -Dcl_khr_local_int32_extended_atomics
     -Dcl_khr_3d_image_writes -Dcl_khr_fp64 -Dcl_khr_int64_base_atomics
     -Dcl_khr_int64_extended_atomics -emit-llvm -target spir64-unknown-unknown
     -o SPIR_OUTPUT.bc -x cl -c SOURCE.cl
     -include <POCL_SOURCE_DIR>/include/_kernel.h
     -include <POCL_SOURCE_DIR>/include/_enable_all_exts.h
