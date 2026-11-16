===================
Metal driver
===================

The Metal driver runs OpenCL kernels on Apple GPUs (Apple silicon Macs) using
the Metal API. It is experimental.

How it works
------------

Metal has no OpenCL C frontend, and its kernel IR -- AIR, the LLVM-bitcode
based IR in Metal libraries -- is undocumented. The driver follows the
approach of `Metal.jl <https://github.com/JuliaGPU/Metal.jl>`_ /
`GPUCompiler.jl <https://github.com/JuliaGPU/GPUCompiler.jl>`_:

* PoCL's Clang compiles the program for the ``spir64`` target, whose address
  space numbering (private 0, global 1, constant 2, local 3) matches AIR's
  (thread, device, constant, threadgroup), and links it with the Metal kernel
  library (``lib/kernel/metal``). The usual SPMD PoCL passes run on the
  result.
* ``lib/CL/devices/metal/pocl-air-gen.cc`` lowers the module to AIR: it
  inlines everything into the kernels, turns the work-item queries into AIR
  kernel input arguments (``thread_position_in_grid`` etc.), passes
  non-pointer arguments by reference through constant buffers, maps LLVM math,
  integer and atomic operations to ``air.*`` functions, and adds the AIR
  kernel and module metadata.
* `libllvm_downgrade <https://github.com/JuliaLLVM/llvm-downgrade>`_ converts
  the bitcode to the LLVM 14 format, as Metal's library loader only reads
  LLVM <= 15 bitcode.
* The driver packages the bitcode in a Metal library (``.metallib``) container
  and loads it with ``-[MTLDevice newLibraryWithData:error:]``. No Xcode or
  Metal toolchain is needed at runtime.

Buffers are ``MTLBuffer`` objects in shared storage; with unified memory the
host accesses them directly. Commands are executed in order by a worker
thread per device, so enqueueing doesn't block.

The generic kernel library implements some math functions with C library
calls. As there's no C library on Metal, the Metal kernel library uses the
libclc-based implementations and SLEEF's portable single-precision code for
these, like the host CPU library does, and CORE-MATH for their ``half``
versions.

Building
--------

Requirements: macOS 15 or newer on Apple silicon, an LLVM supported by PoCL
(with development files), and ``libllvm_downgrade``. Prebuilt, self-contained
binaries are published as
`LLVMDowngrader_jll releases <https://github.com/JuliaBinaryWrappers/LLVMDowngrader_jll.jl/releases>`_
(e.g. ``LLVMDowngrader.v0.12.2.aarch64-apple-darwin.tar.gz``; extract it and
use the extracted directory as ``<downgrade-prefix>``; it needs ``libz`` and
``libzstd`` on the runtime search path). Alternatively, build it from source::

    git clone https://github.com/JuliaLLVM/llvm-downgrade
    cd llvm-downgrade
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
          -DLLVM_DIR=<llvm-prefix>/lib/cmake/llvm \
          -DCMAKE_INSTALL_PREFIX=<downgrade-prefix> -DLLVMDG_BUILD_TESTS=OFF
    cmake --build build && cmake --install build

Then configure PoCL with::

    -DENABLE_METAL=ON -DLLVM_DOWNGRADE_DIR=<downgrade-prefix>

The driver is enabled by default when built; use ``POCL_DEVICES=metal`` to
select only Metal devices, and ``POCL_DEBUG=metal`` for debug output.
``POCL_METAL_DUMP_AIR=1`` (with ``POCL_DEBUG=metal``) prints the lowered AIR
module.

Testing
-------

The tests that are expected to pass on Metal devices have the ``metal``
label::

    POCL_DEVICES=metal ctest -L metal

The tests left out need images, SVM / USM, 64-bit atomics or sub-groups, or
compare printf output that assumes work-groups run one after the other.

Status and limitations
----------------------

* OpenCL 3.0 with program-scope global variables as the only optional
  feature; the OpenCL C version is 3.0 without the generic address space
  (AIR has none).
* ``half`` is supported (``cl_khr_fp16``; disable with
  ``-DENABLE_METAL_FP16=OFF``); ``double`` is not, as Apple GPUs lack it.
* Apple GPUs flush ``float`` denormals to zero (``CL_FP_DENORM`` is not
  reported). Division and ``sqrt`` are correctly rounded
  (``CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT``).
* ``printf`` is supported through a device buffer that the driver decodes
  after each kernel launch (``CL_DEVICE_PRINTF_BUFFER_SIZE`` bytes per
  launch). The output of different work-groups may be interleaved, as they
  run concurrently.
* Program-scope ``__global`` variables live in a per-program buffer that's
  initialized when the program is built (initializers may point to other
  program-scope variables).
* Only 32-bit atomics; ordered atomics are relaxed atomics bracketed by
  fences, as Metal < 4.1 has no ordered atomics.
* ``__local`` pointer kernel arguments are backed by a static threadgroup
  buffer that takes all the threadgroup memory left by the kernels' static
  ``__local`` variables, which can reduce occupancy. Using Metal's
  ``[[threadgroup(n)]]`` arguments needs their (undocumented) AIR metadata.
* Barriers always order both global and local memory.
* No images, samplers, SVM / USM, sub-groups, SPIR-V input, or device-side
  enqueue.
* Kernels may have at most 28 arguments (Metal has 31 buffer bindings; three
  are used for the launch parameters, the printf buffer and the program-scope
  variables).
* Non-uniform work-groups are not supported.
* PoCL inlines everything into the kernels, and Apple's compiler can take
  minutes to compile very large kernels (e.g. the ``kernel/test_as_type``
  test). The compiled libraries are cached in the PoCL kernel cache.

Known workarounds for Apple's back-end
--------------------------------------

* ``__constant`` tables with initializers are marked
  ``externally_initialized``: after unrolling loops over them, Apple's
  compiler folded comparisons on special values (NaN) loaded from them
  incorrectly.
