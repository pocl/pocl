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

A CPU driver turns each kernel into native code by generating a relocatable
object, which it then has to link and load. By default it does this in-process,
with LLVM's ORC LLJIT and the JITLink object-linking layer. JITLink resolves the
object's relocations and maps its code into executable memory, so it acts as
both the linker and the loader. Nothing is exec'd and no shared library is
written; the relocatable object from code generation is the cached artifact
itself, stored with an ``.o`` suffix rather than ``.so`` or ``.dll``.

The alternative is to link the object into a shared library that PoCL loads
with ``dlopen()``. Where lld is available at build time (``CPU_USE_LLD_LINK``),
that link also happens in-process through lld's library API: nothing is exec'd
and no startup files are needed, since the kernel binary's undefined symbols
resolve at ``dlopen()`` time the same way the JIT resolves them. Otherwise the
object is handed to the Clang driver, which needs a host toolchain present at
run time: a linker to exec, plus the C startup files and default libraries.

With in-process lld both modes are thus self-contained; what sets the JIT
apart is not needing the OS loader. The kernels it builds load without
``dlopen()``, so it works with the cache on a ``noexec`` filesystem and in
sandboxes; it resolves symbols against libpocl itself, so it works in
statically linked deployments where a kernel library's references could not
resolve dynamically; and its kernels can be unloaded, where ``dlopen()``\ ed
ones accumulate for the lifetime of the process. The link path is the better
fit where runtime code generation is forbidden but loading shared libraries is
allowed.

The symbols a kernel object refers to still have to come from somewhere. C
library and math functions resolve against the running process, and PoCL
supplies its own host callbacks, such as the printf flush, directly. The
vector-math library chosen at configure time becomes available at startup: the
shared ones (libmvec or SLEEF) load as dynamic libraries, while SVML, which
ships only as static archives, is JIT-linked from ``libsvml.a`` and its
``libirc.a`` helpers, pulling in only the members it references.

The JIT is the default on every platform that supports it: ELF and Mach-O hosts,
and Windows x86-64 (MinGW). It is unavailable on Windows arm64 (JITLink has no
COFF_aarch64 backend) and on MSVC builds, which link through lld in-process
(without the C runtime, against bundled helper objects). To turn the JIT off,
build with ``-DHOST_CPU_ENABLE_JIT=OFF``, or set ``POCL_CPU_JIT=0`` for a single
run (see :ref:`pocl-env-variables`); either one selects the link path.

The JIT does not affect program binaries: a poclbinary exported through
``clGetProgramInfo(CL_PROGRAM_BINARIES)`` contains the linked shared
libraries (any kernel objects only the JIT could load are linked at export
and not serialized), so binaries produced by a JIT-enabled PoCL remain
loadable by builds without the JIT or without LLVM. The kernel cache is
likewise shared between the modes: the loader accepts whichever artifact it
finds, dlopening a shared library even when the JIT is on, and linking a
cached kernel object on the spot when it is off.

With in-process lld this holds even in deployments that ship no host
toolchain at all. Should the export-time link fail anyway, the export falls
back to serializing the kernel objects themselves; such a binary still loads
on any consumer with the JIT enabled (it is JIT-loaded directly) or with LLVM
available (it is linked on import), only a consumer without LLVM cannot use
it.
