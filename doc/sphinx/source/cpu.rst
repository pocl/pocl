**********************
CPU device drivers
**********************

========================
'cpu' driver
========================

This is the default CPU driver, using the pthread library for multithreaded execution.
This driver is the most mature and passes almost entirely the conformance test suite.
Building of this driver is enabled by default, but can be disabled by -DENABLE_HOST_CPU_DEVICES=0.

========================
'cpu-minimal' driver
========================

A minimalistic example CPU device driver for executing kernels on the host CPU. Does not
support multithreading.

========================
'cpu-tbb' driver
========================

This driver uses the Intel Threading Building Blocks (currently named oneTBB) open source library
for work-group and kernel-level task scheduling.

The scheduling characteristics can be fine tuned with environment
variables (see below) to achieve a higher performance.

Building PoCL with TBB
----------------------

1) Install prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

The Intel Threading Building Blocks library must be available on your system.
The location of ``TBBConfig.cmake`` (shipped with TBB since TBB 2017 U7) is
available via ``TBB_DIR`` or ``CMAKE_PREFIX_PATH`` contains path to TBB root.

2) Build PoCL
~~~~~~~~~~~~~

To enable the TBB device, add ``-DENABLE_TBB_DEVICE=1`` to your CMake
configuration command line.

If CMake has trouble locating the TBB library, try specifying the to path to
``TBBConfig.cmake`` by passing ``-DTBB_DIR=<path-to-TBBConfig.cmake>`` to CMake.
Examples::

  -DTBB_DIR=/home/username/intel/tbb_2021/lib/cmake/tbb
  -DTBB_DIR=/home/username/intel/tbb_2020/cmake

3) Configuration
~~~~~~~~~~~~~~~~

When building the tbb device, it will be set as the default device for PoCL instead of pthread driver.
It is strongly recommended to **NOT** create more TBB devices as the TBB device
always uses all cores and has no subdevice support.

The TBB driver can be tuned with at runtime with environment variables, see :ref:`pocl-env-variables`.

.. _cpu-jit:

========================
In-process kernel JIT
========================

A CPU driver compiles each kernel to a relocatable object, then links and loads
it. By default PoCL uses LLVM ORC LLJIT with the JITLink object-linking layer:
the cached final artifact is the object variant (``OBJ_EXT``, ``.so.o`` on
Unix), with no shared library written and no OS loader involved.

With ``POCL_CPU_JIT=0`` or when the JIT is unavailable, PoCL links the object
into a shared library and loads it with ``dlopen()``. If ``CPU_USE_LLD_LINK``
was enabled, this link also runs in-process through lld. Otherwise PoCL invokes
the Clang driver and needs a host linker, startup files, and default libraries.
The link path is useful where runtime code generation is forbidden but shared
libraries are allowed.

JIT symbol resolution is deliberately fixed: process-visible C/libm symbols,
libpocl and its private dependencies, configured vector-math libraries,
compiler-rt/libgcc helpers, selected kernel-library bitcode for late PoCL
builtins, and explicit host ABI callbacks such as printf flush and the MinGW
stack probe. New unresolved symbols should be assigned to one of those classes,
or explained before adding a special-case binding.

The JIT is the default on supported hosts: ELF, Mach-O, and Windows x86-64
(MinGW). Windows arm64 and MSVC builds use the link path. Disable the JIT with
``-DHOST_CPU_ENABLE_JIT=OFF`` at build time or ``POCL_CPU_JIT=0`` for one run.

Exported program binaries remain mode-independent when export-time linking
succeeds: ``clGetProgramInfo(CL_PROGRAM_BINARIES)`` serializes linked shared
libraries, so consumers without JIT or LLVM can load them. For JIT-produced
cache entries, export first links the cached ``OBJ_EXT`` artifacts into shared
libraries and skips those object files. If that link fails, PoCL serializes the
objects as a fallback; those binaries require a JIT-enabled consumer or an
LLVM-enabled non-JIT consumer that can link the objects on import.

The ordinary kernel cache is mode-specific. A JIT device looks for the JIT
object and does not fall back to a sibling shared library. ORC can only use that
file through OS dynamic loading, which is the non-JIT path. On a bad object PoCL
regenerates the object from program IR when possible. A non-JIT device looks for
the shared library, and if only a JIT object is present and LLVM is available,
links it into the shared-library variant before loading.
