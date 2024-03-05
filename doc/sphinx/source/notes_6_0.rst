**************************
Release Notes for PoCL 6.0
**************************



Minimal support for `cl_khr_priority_hints` and `cl_khr_throttle_hints` has been added.
As the extension specification states that these hints provide no guarantees of
any particular behavior (or lack thereof) they are treated as a no-op. However
specifying them no longer causes `clCreateCommandQueueWithProperties` to return
an error.

============================
New device driver: cpu-tbb
============================

The cpu-tbb device driver uses the Intel oneAPI Threading Building Blocks (oneTBB)
library for work-group and kernel-level task scheduling. Except for the
task scheduler, the driver is identical to the original 'cpu' driver (pthread).

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 'cpu' driver gained support for using OpenMP for thread scheduling.
Support is disabled by default, but can be enabled with CMake option. The
'cpu-minimal' driver does not support OpenMP.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Basis for the coarse-grain SVM support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CG SVM support works only if the client manages to mmap() the
device-side allocated SVM pool to the same address as in the
server-side. This is a work-in-progress, but is usable for testing
client apps and libraries that require CG SVM as it seems to work
often enough.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Vsock support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds support for vsock communication to PoCL-Remote. vsock is a
high-performance, low-latency, secure, and scalable network communication
protocol that accelerates guest-host communication in virtualized environments.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
clCompileProgram() and clLinkProgram()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic compile and link support. Tested with conformance suite's
``compiler/test_compiler`` test sets execute_after_simple_compile_and_link,
execute_after_simple_compile_and_link_no_device_info and execute_after_two_file_link
test cases, as well as `chipStar <https://github.com/CHIP-SPV/chipStar>`_,
which uses the API for enhanced SPIR-V portability.

===================================
Deprecation/feature removal notices
===================================

Support for LLVM versions 10 to 13 inclusive has been removed.
LLVM 14 to 17 are supported.

Support for `cl_khr_spir` (SPIR 1.x/2.0) has been removed.
SPIR-V remains supported.
