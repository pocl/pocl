.. _proxy-label:

Proxy driver
=================

This is a driver that is used as proxy to another OpenCL implementation installed on the host.

To build just the proxy driver::

    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LOADABLE_DRIVERS=0 -DENABLE_LLVM=0 -DHOST_DEVICE_BUILD_HASH=<a hash of your choosing> -DENABLE_PROXY_DEVICE=1 -DENABLE_ICD=0 <path-to-pocl-source-dir>

The ``-DHOST_DEVICE_BUILD_HASH`` is required if not using llvm and can be anything, for example ``00000000`` is fine.
``-DENABLE_PROXY_DEVICE_INTEROP=1`` CMake option enables EGL interop on Android, and OpenGL interop otherwise
(requires support from the proxyed implementation).

It's also possible to build with both proxy and another device (for example cpu):

    cmake -DENABLE_HOST_CPU_DEVICES=1 -DENABLE_LOADABLE_DRIVERS=0 -DENABLE_LLVM=1 -DWITH_LLVM_CONFIG=/path/to/llvm-config -DENABLE_PROXY_DEVICE=1 -DENABLE_ICD=0 <path-to-pocl-source-dir>

Changes required to application that wants to use proxy driver:

  * must link to libpocl directly, instead of libOpenCL
  * must ``#include "rename_opencl.h"`` before including any of ``CL/*.h`` headers

The file ``rename_opencl.h`` from pocl has macros that rename (using ``#define``)
all OpenCL API calls to direct pocl calls (clCreateBuffer -> POclCreateBuffer etc).

This is required because the application will have both pocl and libOpenCL linked into it,
but all calls must go through pocl, otherwise crashes are certain. For this reason also,
the proxy driver cannot be built with -DENABLE_ICD=1.

Using LIBOPENCL_STUB
--------------------
Optionally, the proxy driver can also be built with the ``-DPROXY_USE_LIBOPENCL_STUB=1`` option. This will build the proxy driver with the ``libopencl-stub`` library (original code can be found `here <https://github.com/krrishnarraj/libopencl-stub>`_). This has a number of benefits:

    * PoCL does not need to be built as a static library, however you should still link to libpocl and not use the icd loader.
    * You do not need to include ``"rename_opencl.h"`` in your files. In fact, you shouldn't at all.
    * When building PoCL, you do not need to link to an existing OpenCL implementation. This is useful when crosscompiling for Android.
    * During runtime, the Proxy driver will try to dlopen libOpenCL from a number of common locations. These are common locations for MacOS, Windows, Android and Linux. It is also possible to pass a custom location by setting one of the following environment variables during runtime:

        * LIBOPENCL_SO_PATH
        * LIBOPENCL_SO_PATH_2
        * LIBOPENCL_SO_PATH_3
        * LIBOPENCL_SO_PATH_4

Known Issues
------------

    * The proxy driver suffers from the same issues the remote driver has with :ref:`Mali GPUs<remote-issues-label>`. See that section for a workaround.
