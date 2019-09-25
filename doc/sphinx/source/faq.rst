Frequently asked questions
==========================

Common problems and questions related to using and developing pocl
are listed here.

Using pocl
----------

.. _supported-compilers:

Supported compilers and compiler combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pocl usually uses two different compilers (though may be built
using only one). One is used to compile C and C++ files - this is usually
the "system compiler". It's specified by CC and CXX vars to configure
script, or CMAKE_C{,XX}_COMPILER variables to cmake, but usually just
left to default. The second compiler is used to build OpenCL files - this
is always clang+llvm. It's specified by LLVM_CONFIG=<path> to configure,
or -DWITH_LLVM_CONFIG=<path> to cmake.

You may use clang as both "system" and OpenCL compiler for pocl.
Note however that pocl uses the CXX_FLAGS *which the 2nd compiler (clang)
was built with*, to build parts of pocl that link with that compiler. This
may cause some issues, if you try to build pocl with a different compiler
as the one used to build the 2nd compiler - because gcc and clang are not
100% compatible with each other in flags. So far though we've only seen
warnings about unknown flags, not actual bugs.

Anyway, the most trouble-free solution is to use the same "system" compiler
to build pocl, as the one that was used to build the 2nd compiler. Note that
while most Linux distributions use gcc to build their clang/llvm,
the official downloads from llvm.org are built using clang.

Pocl is not listed by clinfo / is not found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Occasionally, proprietary implementations rewrite the ICD loader by their own
version. E.g. Intel SDK installer silently replaces
``/usr/lib/x86_64-linux-gnu/libOpenCL.so`` with a link to
``/etc/alternatives/opencl-libOpenCL.so`` which itself is a link to the intel's
libOpenCL implementation. The fix is to remove the symlinks manually
and reinstall the ICD loader after which both pocl and the Intel SDK
can be used through the ICD loader.

Deadlocks (freezes) on FreeBSD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The issue here is that a library may not initialize the threading on BSD
independently. 
This will cause pocl to stall on some uninitialized internal mutex.
See: http://www.freebsd.org/cgi/query-pr.cgi?pr=163512

A simple work-around is to compile the OpenCL application with "-pthread", 
but this of course cannot be enforced from pocl, especially if an ICD loader 
is used. The internal testsuite works only if "-pthread" is passed 
to ./configure in CFLAGS and CXXFLAGS, even if an ICD loader is used.

clReleaseDevice or clCreateImage missing when linking against -lOpenCL (ICD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions were introduced in OpenCL 1.2. If you have built your ICD
loader against 1.1 headers, you cannot access the pocl implementations of
them because they are missing from the ICD dispatcher.

The solution is to rebuild the ICD loader against OpenCL 1.2 headers.

See: https://github.com/pocl/pocl/issues/27

"Two passes with the same argument (-barriers) attempted to be registered!"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see this error::

  Two passes with the same argument (-barriers) attempted to be registered!
  UNREACHABLE executed at <path..>/include/llvm/Support/PassNameParser.h:73!

It's caused by initializers of static variables (like pocl's LLVM Pass names)
called more than once. This happens for example when you link libpocl twice
to your program.

One way that could happen, is building pocl with ``--disable-icd`` while having
hwloc "plugins" package installed (with the opencl plugin). What happens is:

* libpocl.so gets built, and also libOpenCL.so which is it's copy
* program gets linked to the built libOpenCL.so; that is linked to hwloc
* at runtime, hwloc will try to open the hwloc-opencl plugin; that links to
  system-installed libOpenCL.so (usually the ICD loader);
* the ICD loader will try to dlopen libpocl.so -> you get the error.

The solution is either to use ``--enable-icd --disable-direct-linkage``, or
to uninstall the hwloc "plugins" package.

Why is pocl slow?
^^^^^^^^^^^^^^^^^

If pocl's kernel build seems really slow, it is very possible you have
built your LLVM with Debug+Asserts on (not configure --enable-optimized).
This should result in up to 10x kernel compiler slow downs. You can
really feel it when running 'make check', for example.

The kernel compiler cache often removes that overhead when you
run your OpenCL app the next time.

If pocl is otherwise slower than other OpenCL implementations, it's normal. 
pocl is known to run certain benchmarks faster, certain ones slower, 
when comparing against the Intel and AMD OpenCL SDKs. We hope to improve 
the performance in each release, so if you encounter performance 
regressions (an older pocl/LLVM version used to run an app faster), 
please report a bug.

pocl source code
----------------

Why C99 in host library?
^^^^^^^^^^^^^^^^^^^^^^^^

The kernel compiler passes and some of the driver implementations are in C++11
and it's much faster to implement things in C++11. Why require using C99 in
the host library?

pocl is meant to be very portable to various type of devices, also
to those with very little resources (no operating system at all and with pruned
runtime libraries). C has better portability to low end CPUs and VMs.

Thus, in order for a CPU to act as an OpenCL host without online kernel
compilation support, only C99 support is required from the target,
no C++ compiler, runtime or STL is needed. Also, C programs are said to
sometimes produce more "lightweight" binaries, but that is debatable.
Benchmarks
==============

CLPeak issues
----------------

Currently (Dec 2017) does not work. First, there's a global memory size
detection bug in CLPeak which makes it fail on all OpenCL calls (this
can be workarounded by using POCL_MEMORY_LIMIT=1). Second, compilation
takes forever - this can't be fixed in pocl and needs to be fixed in
either CLPeak or LLVM. CLPeak sources use recursive macros to create
a giant stream of instructions. Certain optimization passes
in LLVM seem to explode exponentially on this code. The second
consequence of giant instruction stream is, it easily overflows the
instruction caches of a CPU, therefore CLPeak results are highly
dependent on whether the compiler manages to fit the code into icache,
perhaps using loop re-rolling, and as such are not a reliable measure
of peak device FLOPS.

Luxmark issues
---------------

* Using the binary downloaded from www.luxmark.info might lead to pocl
  abort on creating cache directory. This is not a bug in Pocl, it's a
  consequence of the two programs (pocl & luxmark) having been compiled
  with different libstdc++. Using a distribution packaged Luxmark
  fixes this problem.

* It's recommended to remove luxmark cache (~/.config/luxrender.net)
  after updating pocl version.

* There's another bug (http://www.luxrender.net/mantis/view.php?id=1640)
  - it crashes after compiling kernels, because it doesn't recognize
  an OpenCL device. This requires editing scenes/<name>/render.cfg,
  you must add ``opencl.cpu.use = 0`` and ``film.opencl.device = 0``

* All scenes (Microphone, Luxball and Hotel) should compile & run
  with LLVM 6 and newer.
