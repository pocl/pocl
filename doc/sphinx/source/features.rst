Known unsupported OpenCL features
=================================

The known unsupported OpenCL (both 1.x and 2.x) features are
listed here as encountered.

Frontend/Clang
--------------

* OpenCL 1.x

  * OpenGL interoperability
  * Image support is incomplete

* OpenCL 2.0

  * generic address space (recognized by LLVM 3.8+ but incomplete)
  * pipes (WIP)
  * device-side enqueue

* cl_khr_f16: half precision float literals

  Compiling "3434.0h" fails with:
  error: invalid suffix 'h' on floating constant

  Tested with Clang 3.4 on 2014-07-10.


Unimplemented host side functions
---------------------------------

The list of unimplemented host-side API functions can be seen as the NULLs in the ICD dispatch struct in
https://github.com/pocl/pocl/blob/master/lib/CL/clGetPlatformIDs.c

