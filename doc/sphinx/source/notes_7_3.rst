**************************
Release Notes for PoCL 7.3
**************************

===========================
Release highlights
===========================

* TBD: Conformance results were submitted for OpenCL 3.0 conformance,
* TBD: Support for LLVM version XX with CUDA, LevelZero and CPU devices
* TBD: Support for LLVM version 24 with CPU device

================
CMake changes
================

==========================
Runtime fixes & features
==========================

* TBD OpenCL-CTS updated to upstream tag v2026-XX-YY-00 and fixed related bugs:

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenASIP (ttasim) driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* FP16 support is now complete and enabled by default on Linux.
  Note that this support has a few requirements:
   - host compiler must support _Float16 (GCC since 12)
   - sufficiently new LLVM which supports _Float16 (since LLVM 19)
   - x86_64, riscv 64 or ARM 64

===================================
Deprecation/feature removal notices
===================================

===================================
Experimental and work-in-progress
===================================

