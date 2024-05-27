**************************
Release Notes for PoCL 6.0
**************************

=======================================================================
New CPU driver which uses Threading Building Blocks for task scheduling
=======================================================================

The 'cpu-tbb' device driver uses the Intel oneAPI Threading Building Blocks (oneTBB)
library for task scheduling. Except for the task scheduler, the driver is identical to
the old 'cpu' driver which uses a custom task scheduler and calls pthreads directly.

===================================================
Experimental cl_ext_buffer_device_address prototype
===================================================

This `new draft extension <https://github.com/KhronosGroup/OpenCL-Docs/pull/1159>`_
prototype enables allocating `cl_mem` buffers with client-accessible
physical addresses which is guaranteed to be fixed for the lifetime of the buffer.
The main difference to coarse-grain SVM allocations is that all SVM allocations require
always the virtual address address to match the device address, thus mapping the buffer
address range also to the vmem even though its host-device transfers were managed only
via explicit memcopies by the application.

Although it's a very simple incremental extension to the basic `clCreateBuffer()` API,
it enables implementing `hipMalloc()` HIP/CUDA and `omp_target_alloc()` OpenMP
allocation calls when the application doesn't require a platform-wide unified address
space.

There is also a prototype implementation of the extension in `Rusticl/Mesa <https://gitlab.freedesktop.org/karolherbst/mesa/-/commit/fa5f51da728dcaf277b0919e90e0400859f290bb>`_.
`chipStar <https://github.com/CHIP-SPV/chipStar>`_ can optionally
use the extension for CUDA/HIP inputs, if neither Unified Shared Memory
(Intel extension) nor OpenCL 2.0+ Coarse-Grain SVM is supported by the
OpenCL device/platform, and the HIP/CUDA application doesn't require unified
address space, but explicitly specifies the memory copy directions.

==========================================
Multi-device command buffer infrastructure
==========================================

Initial support for `cl_khr_command_buffer_multi_device` has been added. It
is now possible to create command buffers associated with multiple command queues
that are not associated with the same device and to remap command buffers to new
(sets of) command queues. The support should be driver-agnostic but has not been
tested with other drivers than CPUs. There likely are no measurable performance
gains from the current implementation either, as everything happens in the
runtime layer of the library.

=====================================
Command queue priority/throttle hints
=====================================

Minimal implementation of `cl_khr_priority_hints` and `cl_khr_throttle_hints` has been
added. As the extension specification states that these hints provide no guarantees of
any particular behavior (or lack thereof) they are treated as a no-op. However
specifying them no longer causes `clCreateCommandQueueWithProperties` to return
an error.

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Support for using OpenMP for task scheduling in the 'cpu' driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenMP is disabled by default, but can be enabled with the CMake
option `ENABLE_HOST_CPU_DEVICES_OPENMP`. The 'cpu-minimal'
driver does not support OpenMP since it's supposed to be a
single-threaded minimal driver.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 * The CPU drivers can be now used for running SYCL programs compiled with the oneAPI binary distributions of DPC++ by adding the following environment settings: `POCL_DRIVER_VERSION_OVERRIDE=2023.16.7.0.21_160000 POCL_CPU_VENDOR_ID_OVERRIDE=32902`.
 * Added support for the `__opencl_c_work_group_collective_functions` feature.
 * Improved SPIR-V support on architectures other than ARM/x86 (like RISC-V).
 * Additional intel_subgroup_shuffle functions (intel_subgroup_block_{read,write}*)
 * Implemented new experimental extensions:

   * cl_pocl_svm_rect: `clEnqueueSVMMemFillRectPOCL` and `clEnqueueSVMMemcpyRectPOCL`. These implement rectangular-region memcpy/memfill with SVM memory.
   * cl_pocl_command_buffer_svm: additional SVM-related commands for use with command buffers, such as `clCommandSVMMemcpyRectPOCL` and `clCommandSVMMemfillRect`
   * cl_pocl_command_buffer_host_buffer: cl_mem & host-memory related commands for use with command buffers, such as `clCommandReadBuffer`, `clCommandReadBufferRect` etc
 * `clGetDeviceAndHostTimer()` implemented.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Basis for the coarse-grain SVM support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CG SVM support works best if the client manages to mmap() the
device-side allocated SVM pool to the same address as in the
server-side. If not, SPIR-V manipulation is done to address for the
offset for kernel executions. This is a work-in-progress, but is usable
for testing client apps and libraries that require CG SVM as it seems to
work often enough.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Vsock support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds support for vsock communication to PoCL-Remote. Vsock is a
high-performance, low-latency, secure, and scalable network communication
protocol that accelerates guest-host communication in virtualized environments.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
clCompileProgram() and clLinkProgram()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic compile and link support. Tested with conformance suite's
``compiler/test_compiler`` test sets `execute_after_simple_compile_and_link`,
`execute_after_simple_compile_and_link_no_device_info` and `execute_after_two_file_link`
test cases, as well as `chipStar <https://github.com/CHIP-SPV/chipStar>`_,
which uses the API for enhancing SPIR-V portability at runtime.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
USM indirect access kernel exec info support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Minimal implementation of the general USM indirect access kernel
execution info flag. It doesn't differentiate between the different
types of USM, but always assumes all USM allocations must be
synchronized when launching a kernel with the general indirect
access flag set.

The buffers-to-synchronize are recorded at enqueue time. That is,
if new USM allocations are added after the enqueue they won't get
synchronized.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Level Zero driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various improvements were made:

* Optimized the host-device synchronization overhead, this should
  be visible mainly with kernels that take less than a millisecond to run

* Implemented support for `ZE_experimental_relaxed_allocation_limits`.
  If the Level Zero driver supports it, PoCL-Level0 will set
  `CL_DEVICE_MAX_MEM_ALLOC_SIZE` to 85% of the available device memory.
  PoCL will automatically compile kernels with both 32bit and 64bit
  pointer offsets, and selects the correct version before execution.

* clLinkProgram() will now use llvm-link instead of spirv-link from
  spirv tools. This is unfortunately necessary because spirv-link does
  not work anymore with files which have different SPIR-V versions.
  spirv-link is not required for building the driver anymore.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various smaller fixes and enhancements, for example:

* Fixed clLinkProgram and clCompileProgram to work correctly
* Fixed memory leaks in clReleaseProgram
* `CL_DEVICE_MAX_MEM_ALLOC_SIZE` limit increased to free memory reported by `cuMemGetInfo`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AlmaIF driver (FPGA interfacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added experimental OpenCL pipe support
* Adds some experimental built-in kernels: sobel, gaussian, phase, magnitude, nonmax suppression and Canny

===================================
Notable fixes
===================================

There were a lot of fixed done over the release cycles. Some of the
most notable/user facing ones are listed below:

* Fixed a buffer overflow when the kernel had SVM/USM indirect pointers.

* libpocl.so is now linked with `--exclude-libs,ALL` linker flag, so
  all imported Clang/LLVM symbols should be hidden if libpocl is linked
  with a statically linked LLVM.

* `clGetDeviceInfo(CL_DEVICE_IL_VERSION)` returns all supported SPIR-V
  versions, not just the latest.

* PoCL is no longer built automatically with LTTNG suppport, it
  needs to be explicitly enabled by a CMake option

* `clWaitForEvents` now calls clFlush before waiting on an event

* Non-versioned binaries of llvm-spirv can be now autodetected
  (their version is checked to match LLVM version)

* New environment variable `POCL_IGNORE_CL_STD=1` will skip
  any ``-cl-std=XY`` option from build options of `clCompileProgram` and `clBuildProgram`.
  This has been found useful when running user programs which supply ``-cl-std=CL2.0``,
  requiring the abundance of features in the OpenCL 2.0, while in fact can run with
  the optional OpenCL 3.0 features implemented by PoCL.

* Support for `clCreateBufferWithPropertiesINTEL` (alias for `clCreateBufferWithProperties`)


===================================
Deprecation/feature removal notices
===================================

 * Support for LLVM versions 10 to 13 inclusive has been removed. LLVM 14 to 18 are supported.
 * Support for `cl_khr_spir` (SPIR 1.x/2.0) has been removed. SPIR-V remains supported.
