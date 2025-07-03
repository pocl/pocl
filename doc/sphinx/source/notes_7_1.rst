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

===========================
Driver-specific features
===========================

* Implemented version 1.0.0 of the cl_khr_spirv_queries extension
  for drivers that support SPIR-V.

===================================
Experimental and work-in-progress
===================================

===================================
Deprecation/feature removal notices
===================================
