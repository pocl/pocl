Proxy driver
=================

This is a driver that is used as proxy to another OpenCL implementation installed on the host.

To build just the proxy driver::

    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_PROXY_DEVICE=1 -DENABLE_ICD=0 <path-to-pocl-source-dir>

``-DENABLE_PROXY_DEVICE_INTEROP=1`` CMake option enables EGL interop on Android, and OpenGL interop otherwise
(requires support from the proxyed implementation).

Changes required to application that wants to use proxy driver:

  * must link to libpocl directly, instead of libOpenCL
  * must ``#include "rename_opencl.h"`` before including any of ``CL/*.h`` headers

The file ``rename_opencl.h`` from pocl has macros that rename (using ``#define``)
all OpenCL API calls to direct pocl calls (clCreateBuffer -> POclCreateBuffer etc).

This is required because the application will have both pocl and libOpenCL linked into it,
but all calls must go through pocl, otherwise crashes are certain. For this reason also,
the proxy driver cannot be built with -DENABLE_ICD=1.
