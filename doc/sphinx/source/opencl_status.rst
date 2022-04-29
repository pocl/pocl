Supported OpenCL features
=========================

All mandatory features for OpenCL 1.2 and 3.0 are supported
on x86-64+Linux, see :ref:`pocl-conformance` for details.

Known unsupported OpenCL features
=================================

The known unsupported OpenCL (both 1.x and 2.x) features are
listed here as encountered.

Frontend/Clang
--------------

* OpenCL 1.x

  * OpenGL interoperability extension
  * SPIR extension (partially available, see below)

* OpenCL 2.0

  * generic address space (recognized by LLVM 3.8+ but incomplete)
  * pipes (WIP)
  * device-side enqueue

* cl_khr_f16: half precision support (with the exception of  vload_half / vstore_half)

Unimplemented host side functions
---------------------------------

All 1.2 API call are implemented. From the 2.x and 3.0 API, all should
exist, but some (e.g. clSetProgramSpecializationConstant) have
"dummy" implementations (they always return an error).

SPIR and SPIR-V support
=========================

There is some experimental support available for SPIR and SPIR-V.
Note that SPIR 1.2 and 2.0 are unsupported (though they may accidentally work);
"SPIR" in the following text refers to LLVM IR bitcode with SPIR target,
the exact format of which is LLVM-version-dependent. The binary format
of SPIR-V is independent of LLVM; for this reason SPIR-V is the preferred format.
Note that SPIR-V format supports different "capabilities" which in effect
are different "dialects" of SPIR-V. The CPU driver supports the "Kernel" dialect,
produced by llvm-spirv, Vulkan driver supports the "Shader" dialect produced
by clspv.

How to build PoCL with SPIR/SPIR-V support
--------------------------------------------

Support for SPIR target is built into LLVM; PoCL built with LLVM automatically supports it.
If you don't require SPIR-V support, you may skip this part.
Support for SPIR-V binaries depends on functional llvm-spirv translator.

Requirements:

* recent PoCL (1.5+ should work)
* recent LLVM (8.0+ works, 7.0 might but is untested)

To compile the LLVM SPIR-V translator::

    git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator
    git checkout <branch>

Check out the corresponding branch for your installed LLVM version, then::

    mkdir build
    cd build
    cmake -DLLVM_DIR=/path/to/LLVM/lib/cmake/llvm ..
    make llvm-spirv

This will produce an executable, ``tools/llvm-spirv/llvm-spirv``. You can copy this executable somewhere,
then when running CMake on PoCL sources, add to the command line: ``-DLLVM_SPIRV=/path/to/llvm-spirv``

Compiling source to SPIR/SPIR-V
--------------------------------

PoCL's own binary format doesn't use SPIR or SPIR-V, but it's possible
to compile OpenCL sources directly to SPIR (LLVM IR with SPIR target),
using Clang::

    clang -Xclang -cl-std=CL1.2 -D__OPENCL_C_VERSION__=120  -D__OPENCL_VERSION__=120 \
     -Dcl_khr_int64 -Dcl_khr_byte_addressable_store -Dcl_khr_int64_extended_atomics \
     -Dcl_khr_global_int32_base_atomics -Dcl_khr_global_int32_extended_atomics \
     -Dcl_khr_local_int32_base_atomics -Dcl_khr_local_int32_extended_atomics \
     -Dcl_khr_3d_image_writes -Dcl_khr_fp64 -Dcl_khr_int64_base_atomics \
     -emit-llvm -target spir64-unknown-unknown \
     -Xclang -finclude-default-header \
     -o SPIR_OUTPUT.bc -x cl -c SOURCE.cl

The SPIR binary from previous command can be further compiled to SPIR-V with::

    llvm-spirv -o SPIRV_OUTPUT.spv SPIR_OUTPUT.bc

Using SPIR/SPIR-V with PoCL
----------------------------

From OpenCL API perspective, PoCL accepts SPIR binaries via  ``clCreateProgramWithBinary`` API.
SPIR-V is accepted only by the ``clCreateProgramWithIL`` API call. This works even
if PoCL only reports OpenCL 1.2 support.

Limitations
-------------

The most complete support is for the CPU device, but there are a few parts
of OpenCL kernel library which CPU driver doesn't yet support with SPIR-V:
Vector datatypes, images, certain geometric math functions.

SPIR / SPIR-V support on other devices than the CPU devices is currently
untested.
