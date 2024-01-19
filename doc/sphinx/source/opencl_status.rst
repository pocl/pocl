Supported OpenCL features
=========================

All mandatory features for OpenCL 1.2 and 3.0 are supported
on x86-64+Linux, see :ref:`pocl-conformance` for details.

Known unsupported OpenCL features
=================================

The known unsupported OpenCL features are listed here as encountered.

Unimplemented device-side features
----------------------------------

* OpenCL 2.0

  * pipes
  * device-side enqueue

* OpenCL 3.0

  see :ref:`pocl-conformance` for the list

Unimplemented host-side features
---------------------------------

All 1.2 runtime API call are implemented. From the 2.x and 3.0 API, all should
exist, but some might have "dummy" implementations (they always return an error).

Unimplemented extensions
------------------------

  * OpenGL interoperability extension
  * DirectX interoperability extension

SPIR-V support
=========================

There is now extensive support available for SPIR-V.

Note that SPIR-V format supports different "capabilities" which in effect
are different "dialects" of SPIR-V. The CPU driver supports the "Kernel" dialect,
produced by llvm-spirv, Vulkan driver supports the "Shader" dialect produced
by clspv.

How to build PoCL with SPIR-V support (CPU / CUDA devices)
----------------------------------------------------------------

Support for SPIR-V binaries depends on functional llvm-spirv translator and some packages.
See :ref:`pocl-install` for additional requirements for SPIR-V support.

Requirements:

* recent PoCL (4.0+ has most extensive support)
* recent LLVM (10.0+ works, for best experience 14+ is recommended)

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

PoCL's own binary format doesn't use SPIR-V, but it's possible
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


Limitations
-------------

The most complete support is for the CPU device on x86-64 and ARM64 platforms,
but there is also some support for CUDA and CPU devices on other platforms.
