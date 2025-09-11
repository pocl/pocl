**************************
Release Notes for PoCL 7.1
**************************

===========================
Release highlights
===========================

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

===========================
Driver-specific features
===========================

* Implemented version 1.0.0 of the cl_khr_spirv_queries extension
  for drivers that support SPIR-V.

===================================
Experimental and work-in-progress
===================================

* Expanded existing defined built-in kernels and introduced a minimal
  set of new ones and implemented them on level0/npu for supporting
  llama.cpp single batch inferences on Intel NPU device. Tested with
  ~1B fp16 parameter variants of Gemma 3.1, Qwen 3 and Llama 3.2
  models using experimental branch
  https://github.com/linehill/llama.cpp/tree/opencl-dbk.

===================================
Deprecation/feature removal notices
===================================
