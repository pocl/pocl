Frequently asked questions
==========================

Common problems and questions related to using and developing pocl
are listed here.

Using pocl
----------

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
run your OpenCL app the next time. It might not always, though. 
For example, currently it doesn't cache kernels which have #include 
clauses in them.

If pocl is otherwise slower than other OpenCL implementations, it's normal. 
pocl is known to run certain benchmarks faster, certain ones slower, 
when comparing against the Intel and AMD OpenCL SDKs. We hope to improve 
the performance in each release, so if you encounter performance 
regressions (an older pocl/LLVM version used to run an app faster), 
please report a bug.
