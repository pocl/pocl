Level Zero driver
=================

This driver uses libze and LLVM/Clang to run OpenCL code on GPU devices via Level Zero API.

The implementation is not yet complete.

Installation
-------------

Required:

 * Level Zero ICD + development files (level-zero and level-zero-devel)
 * Level Zero drivers (on Ubuntu, intel-level-zero-gpu)
 * SPIRV-LLVM-Translator from Khronos (https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
 * SPIR-V tools (in particular, spirv-link)

The ICD + headers must support at least Level Zero specification version 1.3;
older may work but are untested.

To build the Level Zero driver driver::

    cmake -DENABLE_LEVEL0=1 -DENABLE_LLVM=1 -DWITH_LLVM_CONFIG=/path/to/bin/llvm-config <path-to-pocl-source-dir>

After build, libpocl can be tested with (run in the build directory)::

     OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd POCL_BUILDING=1 POCL_DEVICES=level0 ./examples/example1/example1

What's implemented (some were not tested)
-------------------------------------------
 * buffer read/write/map/unmap
 * kernel execution
 * image support (except FillImage)
 * sampler support
 * Spec constants
 * subgroups
 * SVM
 * event timestamps
 * unlimited WG count execution
 * 64bit buffer support (specialization)
 * caching native binaries

Unfinished / non-optimal
-------------------------

 * kernel argument metadata parsing
 *   (type_name is not parsed ATM, the other arg attributes are)
 * CL_MEM_USE_HOST_PTR handling (works with buffers,
 *   but doesn't work with Images)
 * all buffers are allocated using shared memory (zeMemAllocShared),
   this might be a performance problem on dGPUs.
   TODO: investigate & possibly use the virtual + physical memory APIs.

Doesnt work / missing
-----------------------

 * clEnqueueFillImage
 * ZE_MEMORY_ADVICE_SET_READ_MOSTLY optimization
 * support for row_pitch/slice_pitch arguments of Image APIs
 *   ... there are actually two Level0 extension APIs that have the
 *   row/pitch as arguments, but they return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE

Known Bugs
-----------

The FP64 support, on some devices, is software emulated. Kernels using FP64
might not produce results with accuracy that is required by the OpenCL standard.

Testing
---------

The tests that should work with Level Zero driver can be run with tools/scripts/run_level0_tests.

This driver was tested with these devices:

* Intel Tiger Lake integrated GPU
* Intel Alder Lake integrated GPU
