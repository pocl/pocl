
.. _pocl-level0-driver:

Level Zero driver
=================

This driver uses libze and LLVM/Clang to run OpenCL code on GPU devices via Level Zero API.

The implementation is almost complete, there are a few remaining issues.

Installation
-------------

Required:

 * Clang+LLVM: 17, 18 and 19 should work, older may work but are untested.
   It's highly recommended to use a LLVM built with static component libraries (this is the default)
 * Level Zero ICD + development files (level-zero and level-zero-devel)
 * Level Zero drivers (on Ubuntu, intel-level-zero-gpu)
 * SPIRV-LLVM-Translator from Khronos (https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
   Must be built for the corresponding Clang/LLVM branch.
   PoCL will utilize libLLVMSPIRV library if it's available, otherwise
   it will fallback to llvm-spirv binary

Note the driver is known to fail more tests with LLVM 16 because of unsolved
bugs related to opaque-pointer changes in LLVM and SPIRV-LLVM-Translator.

The libze_loader + headers should support at least Level Zero specification version 1.11;
older may work but are untested.

To build the Level Zero driver driver (with recommended options)::

    cmake -DENABLE_LEVEL0=1 -DENABLE_LLVM=1 -DSTATIC_LLVM=ON -DWITH_LLVM_CONFIG=/path/to/bin/llvm-config -DLLVM_SPIRV=/path/to/llvm-spirv <path-to-pocl-source-dir>

Using STATIC_LLVM is recommended to avoid problems with symbol conflicts
(the LevelZero driver uses its own version of LLVM), even though some
LLVM versions might work when linked as shared libraries.

For additional CMake variables see :ref:`pocl-cmake-variables`.

After build, it can be tested without installation (in the build directory)::

    OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd POCL_BUILDING=1 POCL_DEVICES=level0 ./examples/example1/example1

or::
    ./tools/scripts/run_level0_tests

This assumes that `libOpenCL.so` is the opensource ocl-icd loader; for other ICD loaders
you will need to somehow point them to the built `libpocl.so`. For the meaning of environment
variables, see :ref:`pocl-env-variables`.

Known issues / broken features
-------------------------------

These features are currently disabled when PoCL is built with ENABLE_CONFORMANCE=ON.

 * Images: host synchronization when ``CL_MEM_USE_HOST_PTR`` is used works with
   buffers, but doesn't work with Images.
 * Subgroups: require device queries which aren't yet available through L0 API
 * Program-scope variables: fails a particular use case from OpenCL-cts
 * FP64: this is emulated on most consumer hardware, and fails math tests
 * 64bit atomics: can cause a GPU hang

The last two features may work on some hardware but fail on other; the other features
require fixes in software.

Extra features / tunables
--------------------------

``POCL_LEVEL0_JIT`` env variable can be used to enable JIT compilation (kernels are
compiled lazily only when launched via clEnqueueNDRangeKernel, instead of eagerly
at clBuildProgram-time). Useful with programs that have thousands of kernels
(e.g. from heavily templated code). See :ref:`pocl-env-variables` for accepted values.


Testing
---------

The tests that should work with Level Zero driver can be run with ``tools/scripts/run_level0_tests``.

This driver was tested with these de0vices:

* Intel Tiger Lake integrated GPU
* Intel Alder Lake integrated GPU
* Intel Arc A750
