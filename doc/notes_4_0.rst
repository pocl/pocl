
*****************************
Release Notes for PoCL 4.0
*****************************

=============================
Major new features
=============================

-------------------------------------------------------------------------------
PoCL: Support for Clang/LLVM 16.0 (except TCE drivers)
-------------------------------------------------------------------------------

PoCL now supports Clang/LLVM from 10.0 to 16.0 inclusive. The most relevant
change of the new 16.0 release is support for `_Float16 type on x86 and
ARM targets. <https://releases.llvm.org/16.0.0/tools/clang/docs/LanguageExtensions.html#half-precision-floating-point>`_

Clang 16 release notes:
https://releases.llvm.org/16.0.0/tools/clang/docs/ReleaseNotes.html

LLVM 16 release notes:
https://releases.llvm.org/16.0.0/docs/ReleaseNotes.html

-------------------------------------------------------------------------------
CPU driver: support for program-scope variables
-------------------------------------------------------------------------------

Global variables in program-scope are now supported, along with static global
variables in function-scope, for both OpenCL C source and SPIR-V compilation. Passes
the ``basic/test_basic`` test of the OpenCL-CTS, and has been tested with
user applications (CHIP-SPV).

.. code-block:: c

  global float testGlobalVar[128];

  __kernel void test1 (__global const float *a) {
    size_t i = get_global_id(0) % 128;
    testGlobalVar[i] += a[i];
  }

  __kernel void test2 (__global const float *a) {
    size_t i = get_global_id(0) % 128;
    testGlobalVar[i] *= a[i];
  }

  __kernel void test3 (__global float *out) {
    size_t i = get_global_id(0) % 128;
    out[i] = testGlobalVar[i];
  }


-------------------------------------------------------------------------------
CPU driver: support for generic address space
-------------------------------------------------------------------------------

Generic AS is now supported, for both OpenCL C source and SPIR-V compilation.
Passes the ``generic_address_space/test_generic_address_space`` test
of the OpenCL-CTS, and has been tested with user applications (CHIP-SPV).

.. code-block:: c

  int isOdd(int *val) {
    return val[0] % 2;
  }

  __kernel void test3 (__global int *in1, __local int *in2, __global int *out) {
    size_t i = get_global_id(0);
    out[i] = isOdd(in1+i) + isOdd(in2+(i % 128)];
  }

-------------------------------------------------------------------------------
CPU driver: initial(partial) support for cl_khr_subgroups
-------------------------------------------------------------------------------

A single subgroup that always executes the whole X-dimension's WIs.
Independent forward progress is not yet supported, but it's
not needed for CTS compliance, due to the corner case of only one SG in flight.

Additionally, there is partial implementation for ``cl_khr_subgroup_shuffle``,
``cl_intel_subgroups`` and ``cl_khr_subgroup_ballot with caveats``:

  * ``cl_khr_subgroup_shuffle``: Passes the CTS, but only because it doesn't test
    non-uniform(lock-step) behavior, see:
    https://github.com/KhronosGroup/OpenCL-CTS/issues/1236

  * ``cl_khr_subgroup_ballot``: sub_group_ballot() works for uniform calls, the rest
    are unimplemented.

  * ``cl_intel_subgroups``: The block reads/writes are unimplemented.

-------------------------------------------------------------------------------
CPU driver: initial support for cl_intel_required_subgroup_size
-------------------------------------------------------------------------------

This extension allows the programmer to specify the required subgroup size for
a kernel function. This can be important for algorithm correctness. The programmer
can specify the size with a new kernel attribute:
``__attribute__((intel_reqd_sub_group_size(<int>)))``

PoCL additionally implements ``CL_DEVICE_SUB_GROUP_SIZES_INTEL`` parameter for ``clGetDeviceInfo`` API,
however ``CL_​KERNEL_​SPILL_​MEM_​SIZE_​INTEL`` and ``CL_​KERNEL_​COMPILE_​SUB_​GROUP_​SIZE_​INTEL`` for
``clGetKernelWorkGroupInfo`` API are not yet implemented.

-------------------------------------------------------------------------------
CPU driver: partial support for cl_khr_fp16
-------------------------------------------------------------------------------

PoCL now has partial support for ``cl_khr_fp16`` when compiled with Clang/LLVM 16+.
The implementation relies on Clang, and may result in emulation (promoting to
fp32) if the CPU does not support the required instruction set. In
Clang/LLVM 16+, the following targets have native fp16 support: 32-bit and
64-bit ARM (depending on vendor), x86-64 with AVX512-FP16.
Currently only implemented for a part of builtin library functions,
those that are implemented with either an expression, or a Clang builtin.

-------------------------------------------------------------------------------
Level Zero driver: a new device driver, using the Level Zero API.
-------------------------------------------------------------------------------

This driver supports devices accessible via Level Zero API.

The driver has been tested with multiple devices (iGPU and dGPU),
and passes a large portion of PoCL tests (87% tests passed, 32 tests
fail out of 254), however it has not been finished nor optimized yet,
therefore it is not production quality.

The driver supports the following OpenCL extensions, in addition to atomics:
cl_khr_il_program, cl_khr_3d_image_writes,
cl_khr_fp16, cl_khr_fp64, cl_khr_subgroups, cl_intel_unified_shared_memory.
In addition, Specialization Constants and SVM are supported.

We also intend to use the driver for implementing features not found in
the official Intel Compute Runtime OpenCL drivers, and for experimenting
with integration with other OpenCL devices in the same runtime.
One such feature currently implemented is the JIT kernel compilation, which is
useful with programs that have thousands of kernels but only launch a few of
them (e.g. because of templated code).
For details, see the full driver documentation in `doc/sphinx/source/level0.rst`.

-------------------------------------------------------------------------------
CPU driver, Level Zero driver: support for cl_intel_unified_shared_memory
-------------------------------------------------------------------------------

Together with SPIR-V support and other new features, this allows
using PoCL as an OpenCL backend for SYCL runtimes. This works with the
CPU driver (tested on x86-64 & ARM64) and the LevelZero driver. Vincent A. Arcila
has contributed a guide for building PoCL as SYCL runtime backend on ARM.

Additionally, there is a new testsuite integrated into PoCL for testing USM support,
``intel-compute-samples``. These are tests from https://github.com/intel/compute-samples
and PoCL currently passes 78% of the tests (12 tests failed out of 54).

-------------------------------------------------------------------------------
New testsuites
-------------------------------------------------------------------------------

There are also multiple new CTest testsuites in PoCL. For testing PoCL as SYCL backend,
there are three new testsuites: ``dpcpp-book-samples``, ``oneapi-samples`` and ``simple-sycl-samples``.

* ``dpcpp-book-samples``: these are samples from https://github.com/Apress/data-parallel-CPP
  PoCL currently passes 90 out of 95 tests.

* ``oneapi-samples``: these are samples from https://github.com/oneapi-src/oneAPI-samples
  However only a few have been enabled in PoCL for now, because each sample is a separate CMake project

* ``simple-sycl-samples``: these are from https://github.com/bashbaug/simple-sycl-samples
  currently contains only 8 samples, PoCL passes all of them.

For testing PoCL as CHIP-SPV backend: ``chip-spv`` testsuite. This builds
the runtime and the tests from https://github.com/CHIP-SPV/chip-spv, and
runs a subset of tests (approximately 800) with PoCL as OpenCL backend.

-------------------------------------------------------------------------------
Mac OS X support
-------------------------------------------------------------------------------

Thanks to efforts of Isuru Fernando, PoCL has been fixed to work on Mac OS X.
The current 4.0 release has been tested on these configurations:

MacOS 10.13 (Intel Sandybridge), MacOS 11.7 Intel (Ivybridge) with Clang 15

Additionally, there are now Github Actions for CI testing of PoCL with Mac OS X,
testing 4 different configurations: LLVM 15 and 15, with and without ICD loader.

-------------------------------------------------------------------------------
Github Actions
-------------------------------------------------------------------------------

The original CI used by PoCL authors (Python Buildbot, https://buildbot.net)
has been converted to publicly accessible Github Actions CI. These are currently
set up to test PoCL with last two LLVM versions rigorously, and basic tests with
older LLVM versions. The most tested driver is the CPU driver, with multiple
configurations enabling or testing different features: sanitizers, external
testsuites, SYCL support, OpenCL conformance, SPIR-V support. There are also
basic tests for other drivers in PoCL: OpenASIP, Vulkan, CUDA, and LevelZero.

=============================
Bugfixes and minor features
=============================

* CMake: it's now possible to disable libhwloc support even when it's present,
  using -DENABLE_HWLOC=0 CMake option

* AlmaIF's OpenASIP backend now supports a standalone mode.
  It generates a standalone C program from a kernel launch, which
  can then be compiled and executed with ttasim or RTL simulation.

* Added a user env POCL_BITCODE_FINALIZER that can be used to
  call a custom script that manipulates the final bitcode before
  passing it to the code generation.

* New alternative work-group function mode for non-SPMD from Open SYCL:
  Continuation-based synchronization is slightly more general than POCL's
  current kernel compiler, but allows for fewer hand-rolled optimizations.
  CBS is expected to work for kernels that POCL's current kernel compiler
  does not support. Currently, CBS can be manually enabled by setting
  the environment variable `POCL_WORK_GROUP_METHOD=cbs`.

* Linux/x86-64 only: SIGFPE handler has been changed to skip instructions
  causing division-by-zero, only if it occured in one of the CPU driver
  threads; so division-by-zero errors are no longer hidden in user threads.

* CUDA driver: POCL_CUDA_VERIFY_MODULE env variable has been replaced by POCL_LLVM_VERIFY

* CUDA driver: compilation now defaults to `-ffp-contract=fast`, previously it was `-ffp-contract=on`.

* CUDA driver: support for Direct Peer-to-Peer buffer migrations
  This allows much better performance scaling in multi-GPU scenarios

* OpenCL C: `-cl-fast-relaxed-math` now defaults to `-ffp-contract=fast`, previously it was `-ffp-contract=on`.

* CPU drivers: renamed 'basic' to 'cpu-minimal' and 'pthread' driver to 'cpu',
  to reflect the hardware they're driving instead of implementation details.

* CPU drivers: POCL_MAX_PTHREAD_COUNT renamed to POCL_CPU_MAX_CU_COUNT;
  the old env. variable is deprecated but still works

* CPU drivers: Added a new POCL_CPU_LOCAL_MEM_SIZE environment for overriding the
  local memory size.

* CPU drivers: OpenCL C printf() flushes output after each call instead of waiting
  for the end of the kernel command. This makes it more useful for debugging
  kernel segfaults.

