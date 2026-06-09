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

By default a CPU driver turns each kernel into native code by handing the
compiled object to the Clang driver, which links it into a shared library that
PoCL then loads with ``dlopen()``. The link step needs a host toolchain present
at run time: a linker to exec, plus the C startup files and default libraries.

Built with ``-DHOST_CPU_ENABLE_JIT=ON`` (the default where it is supported), the
CPU drivers load the kernel object in-process instead, using LLVM's ORC LLJIT
with the JITLink object-linking layer. JITLink resolves the object's relocations
and maps its code into executable memory, so it stands in for both the linker
and the loader. Nothing is exec'd and no shared library is written: the
relocatable object from code generation is the cached artifact itself, stored
with an ``.o`` suffix rather than ``.so`` or ``.dll``. Removing the host linker
and the startup files is what makes it practical to ship PoCL to a machine with
no development environment.

The symbols a kernel object refers to still have to come from somewhere. C
library and math functions are resolved from the running process. PoCL's own
host callbacks, such as the printf flush, are supplied directly. The vector-math
library chosen at configure time (libmvec or SLEEF) is loaded once at startup.
SVML is the exception: it ships only as a static library, so SVML builds keep
the Clang-driver link path and leave ``HOST_CPU_ENABLE_JIT`` off.

The JIT covers ELF and Mach-O hosts and Windows x86-64 (MinGW), and needs LLVM
18 or newer; on older LLVM the drivers keep the Clang-driver link path. Turn it
off at build time with ``-DHOST_CPU_ENABLE_JIT=OFF``, or for a single run with
``POCL_CPU_JIT=0`` (see :ref:`pocl-env-variables`); either one selects the
Clang-driver link path.
