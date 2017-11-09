Known unsupported OpenCL features
=================================

The known unsupported OpenCL (both 1.x and 2.x) features are
listed here as encountered.

Frontend/Clang
--------------

* OpenCL 1.x

  * OpenGL interoperability extension
  * SPIR extension

* OpenCL 2.0

  * generic address space (recognized by LLVM 3.8+ but incomplete)
  * pipes (WIP)
  * device-side enqueue

* cl_khr_f16: half precision float literals


Unimplemented host side functions
---------------------------------

All 1.2 API call should be implemented. The list of unimplemented
2.0 calls can be seen as the NULLs in the ICD dispatch struct in
https://github.com/pocl/pocl/blob/master/lib/CL/clGetPlatformIDs.c

