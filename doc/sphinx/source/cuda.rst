==================
NVIDIA GPU support
==================

NOTE: Support for NVIDIA GPUs via the CUDA backend is currently experimental
and many features may be missing or incomplete.

The experimental CUDA backend provides support for CUDA-capable NVIDIA GPUs
under Linux or macOS.
The goal of this backend is to provide an open-source alternative to the
proprietary NVIDIA OpenCL implementation.
This makes use of the NVPTX backend in LLVM and the CUDA driver API.

Building PoCL with CUDA support
-------------------------------

1) Install prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~
  Aside from the usual PoCL dependencies, you will also need the CUDA toolkit.
  Currently the backend is only regularly tested against CUDA 11.6, but it should
  be possible to build against other versions.

  If you experience build failures regarding missing CUDA headers or libraries,
  you may need to add the include directory containing ``cuda.h`` to your header
  search path, and/or the library directory containing ``libcuda.{so,dylib}`` to
  your library search path.

  The CUDA backend requires LLVM built with the NVPTX backend enabled.

2) Build PoCL
~~~~~~~~~~~~~
  To enable the CUDA backend, add ``-DENABLE_CUDA=ON`` to your CMake
  configuration command line.

  Otherwise, build and install PoCL as normal.

3) Run tests
~~~~~~~~~~~~
  After building PoCL, you can smoke test the CUDA backend by executing the
  subset of PoCL's tests that are known to pass on NVIDIA GPUs::

    ../tools/scripts/run_cuda_tests

4) Configuration
~~~~~~~~~~~~~~~~
  Use ``POCL_DEVICES=CUDA`` to select only CUDA devices. If the system has more
  than one GPU, specify the ``CUDA`` device multiple times (e.g.
  ``POCL_DEVICES=CUDA,CUDA`` for two GPUs).

  The CUDA backend currently has a runtime dependency on the CUDA toolkit. If
  you receive errors regarding a failure to load ``libdevice``, you may need
  to set the ``POCL_CUDA_TOOLKIT_PATH`` environment variable to tell PoCL
  where the CUDA toolkit is installed.
  Set this variable to the root of the toolkit installation (the directory
  containing the ``nvvm`` directory).

  The ``POCL_CUDA_GPU_ARCH`` environment variable can be set to override the
  target GPU architecture (e.g. ``POCL_CUDA_GPU_ARCH=sm_35``), which may be
  necessary in cases where LLVM doesn't yet support the architecture.

  The ``POCL_CUDA_DUMP_NVVM`` environment variable can be set to ``1`` to
  dump the LLVM IR that is fed into the NVPTX backend for debugging purposes
  (requires ``POCL_DEBUG=1``).

  The ``POCL_CUDA_DISABLE_QUEUE_THREADS`` environment variable can be set to
  ``1`` to disable background threads for handling command submission. This can
  potentially reduce command launch latency, but can cause problems if using
  user events or sharing a context with a non-CUDA device.

CUDA backend status
-------------------

(last updated: 2017-06-02)

The CUDA backend currently passes 73 tests from PoCL's internal testsuite, and
is capable of running various real OpenCL codes.
Unlike NVIDIA's proprietary OpenCL implementation, PoCL supports SPIR-V
consumption, and so this backend has also been able to run (for example) SYCL
codes using Codeplay's ComputeCpp implementation on NVIDIA GPUs.
Since it uses CUDA under-the-hood, this backend also works with all of the
NVIDIA CUDA profiling and debugging tools, many of which don't work with
NVIDIA's own OpenCL implementation.

Conformance status
~~~~~~~~~~~~~~~~~~

The Khronos OpenCL 1.2 conformance tests are
`available here <https://github.com/KhronosGroup/OpenCL-CTS/tree/cl12_trunk>`_.
The following test categories are known to pass on at least one NVIDIA GPU using
PoCL's CUDA backend:

* allocations
* api
* atomics
* basic
* commonfns
* computeinfo
* contractions
* events
* profiling
* relationals
* thread_dimensions
* vec_step

Tested platforms
~~~~~~~~~~~~~~~~
The CUDA backend has been tested on Linux (CentOS 7.3) with SM_35, SM_52,
SM_60, and SM_61 capable NVIDIA GPUs.

The backend is also functional on macOS, with just one additional test failure
compared to Linux (``test_event_cycle``).

Known issues
~~~~~~~~~~~~
The following is a non-comprehensive list of known issues in the CUDA backend:

* image types and samplers are unimplemented
* printf format support is incomplete

Additionally, there has been little effort to optimize the performance of this
backend so far - the current effort is on implementing remaining functionality.
Once the core functionality is completed, optimization of the code generation
and runtime can begin.

Support
~~~~~~~
For bug reports and questions, please use PoCL's `GitHub issue tracker
<https://github.com/pocl/pocl/issues>`_.
Pull requests and other contributions are also very welcome.

This work has primarily been done by James Price from the
`University of Bristol's High Performance Computing Group
<http://uob-hpc.github.io>`_.
