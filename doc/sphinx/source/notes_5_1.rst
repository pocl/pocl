**************************
Release Notes for PoCL 5.1
**************************

Support for LLVM versions 10 to 13 inclusive has been removed.
LLVM 14 to 17 are supported.

Support for  `cl_khr_spir` (SPIR 1.x/2.0) has been removed.
SPIR-V remains supported.

========================
New device driver: TBB
========================

The TBB device driver uses the Intel oneAPI Threading Building Blocks (oneTBB)
library for work-group and kernel-level task scheduling. Except for the
scheduling, the driver is identical to the original CPU driver (pthread).

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CPU driver gained support for OpenMP scheduling. Support is
disabled by default, but can be enabled with CMake option. The
cpu-minimal driver does not support OpenMP and remains single-threaded.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote: Basis for the coarse-grain SVM support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CG SVM support works only if the client manages to mmap() the
device-side allocated SVM pool to the same address as in the
server-side. This is a work-in-progress, but is usable for testing
client apps and libraries that require CG SVM as it seems to work
often enough.

===================================
Deprecation/feature removal notices
===================================
