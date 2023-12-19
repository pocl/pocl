
.. _pocl-level0-driver:

Level Zero driver
=================

This driver uses libze and LLVM/Clang to run OpenCL code on GPU devices via Level Zero API.

The implementation is work-in-progress, but usable for various applications.

Installation
-------------

Required:

 * Clang+LLVM: 14 and 15 should work, older may work but are untested
 * Level Zero ICD + development files (level-zero and level-zero-devel)
 * Level Zero drivers (on Ubuntu, intel-level-zero-gpu)
 * SPIRV-LLVM-Translator from Khronos (https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
   Must be built for the corresponding Clang/LLVM branch.
   Preferably the `llvm-spirv` binary should be in the same path as `llvm-config`,
   otherwise PoCL's CMake could pick up a different (wrong) `llvm-spirv`.
 * SPIR-V tools (in particular, `spirv-link`)

The ICD + headers must support at least Level Zero specification version 1.3;
older may work but are untested.

To build the Level Zero driver driver::

    cmake -DENABLE_LEVEL0=1 -DENABLE_LLVM=1 -DWITH_LLVM_CONFIG=/path/to/bin/llvm-config <path-to-pocl-source-dir>

For additional CMake variables see :ref:`pocl-cmake-variables`.

After build, it can be tested without installation (in the build directory)::

    OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd POCL_BUILDING=1 POCL_DEVICES=level0 ./examples/example1/example1

This assumes that `libOpenCL.so` is the opensource ocl-icd loader; for other ICD loaders
you will need to somehow point them to the built `libpocl.so`. For the meaning of environment
variables, see :ref:`pocl-env-variables`.

What's implemented (some were not tested)
-------------------------------------------
 * buffer read/write/map/unmap
 * kernel execution
 * image support
 * sampler support
 * Spec constants
 * subgroups
 * SVM
 * 64bit buffer support (specialization)
 * caching native binaries

Unfinished / non-optimal
-------------------------

 * kernel argument metadata parsing (``type_name`` is
   not parsed ATM, the other argument attributes are)
 * host synchronization when ``CL_MEM_USE_HOST_PTR`` is used (works with
   buffers, but doesn't work with Images)
 * all buffers are allocated using shared memory (``zeMemAllocShared``),
   this might be a performance problem on dGPUs.
   TODO: investigate & possibly use the virtual + physical memory APIs.

Doesnt work / missing
-----------------------

 * ``ZE_MEMORY_ADVICE_SET_READ_MOSTLY`` optimization
 * support for row_pitch/slice_pitch arguments of Image APIs
   ... there are actually two Level0 extension APIs that have the row/pitch arguments,
   but they currently return ``ZE_RESULT_ERROR_UNSUPPORTED_FEATURE``

Extra features / tunables
--------------------------

``POCL_LEVEL0_JIT`` env variable can be used to enable JIT compilation (kernels are
compiled lazily only when launched via clEnqueueNDRangeKernel, instead of eagerly
at clBuildProgram-time). Useful with programs that have thousands of kernels
(e.g. from heavily templated code). See :ref:`pocl-env-variables` for accepted values.

Known Bugs
-----------

The FP64 support, on some devices, is software emulated. Kernels using FP64
might not produce results with accuracy that is required by the OpenCL standard.

Certain tests may pass on some GPUs but not others. Also, the driver is known to fail more
tests with LLVM 16 because of unsolved bugs related to opaque-pointer changes in LLVM and SPIRV-LLVM-Translator.

Testing
---------

The tests that should work with Level Zero driver can be run with ``tools/scripts/run_level0_tests``.

This driver was tested with these de0vices:

* Intel Tiger Lake integrated GPU
* Intel Alder Lake integrated GPU
* Intel Arc A750
