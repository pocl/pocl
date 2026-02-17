**************************
Release Notes for PoCL 7.2
**************************

================
CMake changes
================

* PoCL now supports configuring LLVM via both llvm-config binary (`-DWITH_LLVM_CONFIG=<path>`),
  and LLVMConfig.cmake (`-DLLVM_DIR=<path>`). Support for cross-compiling was also improved,
  and requires using both these options, pointing to target & host LLVMs.

* `LLVM_SPIRV` CMake option was replaced by `HOST_LLVM_SPIRV`

================
Notable bugfixes
================

* Fixed various clLinkProgram issues in the remote driver.
* Fixed remote driver spuriously reconnecting for no apparent reason.
* OpenCL-CTS updated to 2026.01.27 and fixed newly uncovered bugs
  (some corner-cases in clSetKernelArg, clSetKernelExecInfo, clCreateContext,
   clCreateContextFromType, clGetDeviceIDs, clSetKernelArgSVMPointer,
   clEnqueueNDRange with local_size == NULL and nonzero reqq-wg-size)

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Implement sub_group_{reduce,scan_exclusive,scan_inclusive}_* and
  sub_group_{all,any,broadcast}. (reduce is only available for PTX6.0+)
* Note that CUDA driver does not support LLVM 21, due to a bug
  in upstream Clang code. Users must use LLVM 18 to 20 with CUDA. For details,
  see https://github.com/llvm/llvm-project/issues/154772

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenASIP (ttasim) driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Upgraded the driver to use OpenASIP v2.2(-pre) which uses LLVM 21.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The atomic min/max operation on floats (cl_ext_float_atomic) are now
  disabled when `-DENABLE_CONFORMANCE=ON` ; atomic sub/add are enabled.
  When `-DENABLE_CONFORMANCE=OFF`, all atomics on floats are enabled.

* Implemented new extensions: cl_khr_extended_bit_ops, cl_khr_device_uuid,
  cl_khr_suggested_local_work_size, cl_khr_integer_dot_product

===================================
Deprecation/feature removal notices
===================================

* Support for LLVM version 17 was removed, versions 18 to 21 are supported

===================================
Experimental and work-in-progress
===================================

* Added a work-in-progress MLIR-based OpenCL C kernel compilation flow for CPUs.
  The flow supports both Polygeist and ClangIR front-ends.
  Support for basic features such as local memory and some built-ins is included,
  but the majority of built-in functions or barriers are not yet supported.
  Contributions are welcome to increase the test coverage.
