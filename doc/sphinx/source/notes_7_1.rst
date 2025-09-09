**************************
Release Notes for PoCL 7.1
**************************

This is mostly a bug-fix/maintenance release after the large 7.0 one.

===========================
Release highlights
===========================

* Support for LLVM 21 for the CPU devices.

* Support for cl_khr_icd v2.0.0, cl_khr_spirv_queries and SPV_KHR_expect_assume.

* Various stability and ease-of-setup improvements on the Windows port, for
  example no longer requiring MS Visual Studio Build Tools for linking
  CPU device kernels.

=============================
Notable user facing changes
=============================

* Improved overhead of clEnqueueNDRange() calls in cases where there
  are several hundreds of SVM/USM allocations. For example, on
  OpenVINO running resnet50 inference, the call time reduced to few
  microseconds from previous ~20us.

* Improved error message when a recursive function is encountered:
  Print the infringing function in addition to the function where the recursion
  was encountered and demangle C++ function names.

* Windows builds no longer require MS Visual Studio Build Tools for linking
  CPU device kernels. This only works with 1) static LLVM built with lld-link,
  2) PoCL built with MSVC compiler for x86(-64) target. The only remaining
  runtime dependency is the MSVC runtime library.

================
Notable bugfixes
================

* Multiple fixes on the fine-grain sub-buffer migration code.

===========================
Driver-specific features
===========================

* Implemented version 1.0.0 of the cl_khr_spirv_queries extension
  for drivers that support SPIR-V.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Report SPIR-V 1.3 and 1.4 support when using LLVM 18.
* Support SPV_KHR_expect_assume.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Level Zero driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Various bugfixes.
* Enable SPV_INTEL_memory_access_aliasing.

===================================
Experimental and work-in-progress
===================================

===================================
Deprecation/feature removal notices
===================================

* Support for LLVM versions older than 17.0 was removed.
