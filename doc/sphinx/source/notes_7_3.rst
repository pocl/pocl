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

* Added an `ENABLE_CUDA_IMAGES` option. Note that image support in the CUDA
  driver is still experimental and very incomplete.

==========================
Runtime fixes & features
==========================

* TBD: OpenCL-CTS updated to upstream tag v20XX-YY-ZZ-00 and fixed related bugs:

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Initial bits for images support. Currently only the `IMAGE1D_BUFFER` image
  type is supported.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Barriers and markers are now handled server-side.
* Various server-side command buffer implementation fixes
* Remote now always advertises `cl_khr_command_buffer` even if the server-side
  OpenCL driver does not (pocld provides server-side emulation in that case).

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
   - x86_64, RISC-V 64 or ARM 64

===================================
Deprecation/feature removal notices
===================================

===================================
Experimental and work-in-progress
===================================
