**************************
Release Notes for PoCL 7.2
**************************

===========================
Release highlights
===========================

* Conformance results were submitted for OpenCL 3.0 conformance with the
  `CPU x86-64 (AVX512) <https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_468>`_ ,
  `CPU RISC-V RV64GC <https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_469>`_ and
  `CPU RISC-V RVA22+RVV1.0 <https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_469>`_
  devices, using OpenCL-CTS tag v2026-03-25-00.
  Conformance testing via OpenCL-CTS passed with both OpenCL C
  and SPIR-V compilation modes, for all of these devices.

* Support for LLVM version 22

================
CMake changes
================

* PoCL now supports configuring LLVM via both llvm-config binary (`-DWITH_LLVM_CONFIG=<path>`),
  and LLVMConfig.cmake (`-DLLVM_DIR=<path>`). Support for cross-compiling was also improved,
  and requires using both these options, pointing to target & host LLVMs.

* `LLVM_SPIRV` CMake option was replaced by `HOST_LLVM_SPIRV`

* Added `POCL_DEBUG_LLVM_OPTS` environment variable. This allows directly
  passing multiple LLVM options for debugging purposes.

==========================
Runtime fixes & features
==========================

* OpenCL-CTS updated to upstream tag v2026-03-25-00 and fixed related bugs:

  * clCreateContext: return error on invalid `CL_CONTEXT_INTEROP_USER_SYNC` flag
  * clSetKernelArg: for buffer argument, accept arg_value NULL or pointer to NULL
  * clSetKernelExecInfo: fix out-of-bounds read of the USM indirect pointer list.
  * clCreateContextFromType: return error on unknown values of device_type bitfield
  * clGetDeviceIDs: return error on unknown values of device_type bitfield
  * clSetKernelArgSVMPointer: return `CL_INVALID_ARG_INDEX` for some error conditions
  * clEnqueueNDRange: accept `local_size == NULL` if required-wg-size is nonzero
  * clCreateProgramWithIL: return `CL_INVALID_OPERATION` if no devices support SPIR-V

* Fixed an out-of-bounds read in `clSetKernelExecInfo` when registering
  indirect USM pointers (`CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL`): the size of
  the pointer list was passed on in bytes where a pointer count was expected,
  causing the runtime to read past the end of the user-provided list.

* `clGetKernelArgInfo(CL_KERNEL_ARG_NAME)` now returns the source
  argument names for SPIR/SPIR-V binaries on the CPU driver (previously
  returned `CL_KERNEL_ARG_INFO_NOT_AVAILABLE` even when the binary
  contained the names). The `llvm-spirv` translator preserves SPIR-V
  `OpName` decorations on `llvm::Argument`s rather than as
  `!kernel_arg_name` metadata, and PoCL now reads them from there.

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed various clLinkProgram issues in the remote driver.

* Fixed remote driver spuriously reconnecting for no apparent reason.

* Added support for event tracing of network commands

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Implemented `sub_group_{reduce,scan_exclusive,scan_inclusive}_*` and
  `sub_group_{all,any,broadcast}`. (reduce is only available for PTX6.0+)

* Note that CUDA driver does not support LLVM 21, due to a bug
  in upstream Clang code. Users must use LLVM 18,19,20 or 22 with CUDA.
  For details see https://github.com/llvm/llvm-project/issues/154772

* Fixed pre-mangled subgroup intrinsic names

* Implemented support for additional parameters to clGetKernelSubgroupInfo():
  `CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE`,
  `CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE`,
  `CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT`

* Fixed `printf` to work correctly when using LLVM 19/20 (issue 2021)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenASIP (ttasim) driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Upgraded the driver to use OpenASIP v2.2 which uses LLVM 21.

* Removed the big endian target support

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The atomic operations on floats (`cl_ext_float_atomic`) are now
  disabled when `-DENABLE_CONFORMANCE=ON`.
  When `-DENABLE_CONFORMANCE=OFF`, `cl_ext_float_atomic` is enabled.
  This is due to issues in OpenCL-CTS that prevent passing the testsuite
  when `cl_ext_float_atomic` is enabled.

* Implemented new extensions: `cl_khr_extended_bit_ops`, `cl_khr_device_uuid`,
  `cl_khr_suggested_local_work_size`, `cl_khr_integer_dot_product`

* Improved support for FP16 math builtins. Not all of the math builtins
  are supported yet; approx 15 are still missing.

* When using SPIR-V input, extensions `cl_khr_spirv_linkonce_odr` and
  `cl_khr_spirv_no_integer_wrap_decoration` are now supported

* Fixed missing support of a few FP16 overloads
  of `vload/vstore/convert_type` builtins in SPIRV input

* Reported sizes for `INTEL_SUB_GROUP_SIZES` were reverted to earlier behavior,
  PoCL now only reports power-of-2 sizes.

* Implemented support for `cl_khr_kernel_clock` (requires Clang support for
  `__builtin_readcyclecounter` builtin function)

* Re-enabled support for `__opencl_c_program_scope_global_variables`. This had
  a problem with SPIR-V input when using certain versions of `llvm-spirv`
  translator, however this has now been fixed in the translator.

* Mac OS X: statically-linked LLVM symbols are now hidden also when using the CMake-package setup

* Fixed kernel-side sub-group queries for non-divisible work-group sizes

* Honor `intel_reqd_sub_group_size` attribute in clGetKernelSubGroupInfo()

* `SIGFPE` handler is now fixed to enable only on x86-64, not all x86

===================================
Deprecation/feature removal notices
===================================

* Support for LLVM version 17 was removed, versions 18 to 22 are supported

===================================
Experimental and work-in-progress
===================================

* Added a work-in-progress MLIR-based OpenCL C kernel compilation flow for CPUs.
  The flow supports both Polygeist and ClangIR front-ends.
  Support for basic features such as local memory and some built-ins is included,
  but the majority of built-in functions or barriers are not yet supported.
  Contributions are welcome to increase the test coverage.
