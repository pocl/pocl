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

