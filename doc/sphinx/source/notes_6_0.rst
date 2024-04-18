**************************
Release Notes for PoCL 6.0
**************************

============================
New device driver: cpu-tbb
============================

The cpu-tbb device driver uses the Intel oneAPI Threading Building Blocks (oneTBB)
library for work-group and kernel-level task scheduling. Except for the
task scheduler, the driver is identical to the original 'cpu' driver (pthread).

=====================================
Command queue priority/throttle hints
=====================================

Minimal support for `cl_khr_priority_hints` and `cl_khr_throttle_hints` has been added.
As the extension specification states that these hints provide no guarantees of
any particular behavior (or lack thereof) they are treated as a no-op. However
specifying them no longer causes `clCreateCommandQueueWithProperties` to return
an error.

===================================================
Experimental cl_ext_buffer_device_address prototype
===================================================

This new extension prototype enables allocating `cl_mem` buffers with client-accessible
physical addresses which is guaranteed to be fixed for the lifetime of the buffer.
The main difference to coarse-grain SVM allocations is that all
SVM allocations require always the virtual address address to match the device address,
thus mapping the buffer address range also to the vmem even though its contents
are managed only via explicit memcopies by the application.

Although it's a very simple incremental extension to the basic `clCreateBuffer()` API,
it enables implementing `hipMalloc()` HIP/CUDA and `omp_target_alloc()` OpenMP
allocation calls when the application doesn't require a unified address space.

There is also a prototype implementation of the extension in `Rusticl/Mesa <https://gitlab.freedesktop.org/karolherbst/mesa/-/commit/fa5f51da728dcaf277b0919e90e0400859f290bb>`_.

`chipStar <https://github.com/CHIP-SPV/chipStar>`_ can optionally
use the extension, if neither Unified Shared Memory (Intel extension) nor
OpenCL 2.0+ Coarse-Grain SVM is supported by the OpenCL device/platform,
and the HIP/CUDA application doesn't require unified address space, but
explicitly specifies the memory copy directions.

The actual extension text is yet to write and the extension can
change without notification as we get feedback and more experience from
using it.

============================
Multi-device command buffers
============================

Initial support for `cl_khr_command_buffer_multi_device` has been added, i.e. it
is now possible to create command buffers associated with multiple command queues
that are not associated with the same device and to remap command buffers to new
(sets of) command queues. This should be driver-agnostic but has not been tested
with other drivers than CPU. There likely are no large performance gains from
the current implementation either, as everything happens in the surface layers
of the library.

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * Support for using OpenMP for task scheduling was added. It is disabled
   by default, but can be enabled with CMake option. The 'cpu-minimal'
   driver does not support OpenMP since it's supposed to be single-threaded.
 * The CPU drivers can be now used for running SYCL programs compiled with
   the oneAPI binary distributions of DPC++ by adding the following environment
   settings: **POCL_DRIVER_VERSION_OVERRIDE=2023.16.7.0.21_160000 POCL_CPU_VENDOR_ID_OVERRIDE=32902**.
 * Added support for the **__opencl_c_work_group_collective_functions** feature.
 * improved SPIR-V support on architectures other than ARM/x86 (like RISC-V)

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
