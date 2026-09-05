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
* `ENABLE_HOST_CPU_VECTORIZE_LIBMVEC` and `ENABLE_HOST_CPU_VECTORIZE_SLEEF`
  can now be combined with `ENABLE_CONFORMANCE` (default OFF there, opt-in).
  `ENABLE_HOST_CPU_VECTORIZE_SVML` still cannot.

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
* Remote command handling has been reworked to avoid blocking remote commands
  in the client's submit queue and pocld's reply queue.
* Fixed some issues where one of the client-server sockets would abruptly
  disconnect for no obvious reason.
* Server-to-server completion notifications are elided when possible to reduce
  overhead in setups with a large number of servers in the same context
* Server-to-server buffer migrations are now properly asynchronous
* Fixes for compiling kernels for multiple devices
* Fixed fetching program binaries and recreating programs from previously built
  binaries

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenASIP (ttasim) driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Vectorized math libraries (libmvec, SLEEF) are now filtered per function:
  builtins whose library implementation was measured to exceed the OpenCL C
  ULP bounds keep their scalar implementation (`pocl_vecmath_deny.h`,
  generated from one list in CMake). This fixes the default non-conformance
  build failing the CTS `math_brute_force` test for float `log` at vector
  width 1 and double `exp` at one argument (glibc libmvec), and double
  `pow` near the overflow threshold (SLEEF older than 3.8). `POCL_VECMATH_DENY` / `POCL_VECMATH_ALLOW` override
  the list for measurements.
* The Clang-builtin swap in the CPU kernel library now follows the selected
  vector library: only builtins with a vector implementation are swapped,
  the rest keep their SLEEF or libclc sources. `pown` is no longer swapped
  (it went through `powi`: hundreds of ULP over the bound in the CTS,
  millions in a wider sweep).
* On x86-64 with glibc 2.35 or newer the whole libmvec function set is
  used (26 functions instead of the 6 in LLVM's table). The 20 added
  functions run at the memory-bandwidth floor in scalar-typed kernels,
  from 2 to 10 ns per element before. AVX-512 variants are only used on
  hosts with AVX-512.

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
