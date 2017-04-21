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

Building pocl with CUDA support
-------------------------------

1) Install prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~
  Aside from the usual pocl dependencies, you will also need the CUDA toolkit.
  Currently this backend has only been tested against CUDA 8.0, but it may also
  be possible to build against other versions.
  The include directory containing ``cuda.h`` should be on your header search
  path, and the library directory containing ``libcuda.{so,dylib}`` should be
  on your library search path.

  The CUDA backend requires LLVM 4.0 or newer, and LLVM must have been built
  with the NVPTX backend enabled.

2) Build pocl
~~~~~~~~~~~~~
  To enable the CUDA backend, add ``-DENABLE_CUDA=ON`` to your CMake
  configuration command line.

  Otherwise, build and install pocl as normal.

3) Configuration
~~~~~~~~~~~~~~~~
  The CUDA backend currently has a runtime dependency on the CUDA toolkit.
  The ``POCL_CUDA_TOOLKIT_PATH`` environment variable needs to be set to tell
  pocl where the CUDA toolkit is installed.
  Set this variable to the root of the toolkit installation (the directory
  containing the ``nvvm`` directory).

4) Run tests
~~~~~~~~~~~~
  After building pocl, you can smoke test the CUDA backend by executing the
  subset of pocl's tests that are known to pass on NVIDIA GPUs::

    ../tools/scripts/run_cuda_tests


CUDA backend status
-------------------

(last updated: 2017-04-21)

The CUDA backend currently passes 74 tests from pocl's internal testsuite, and
is capable of running various real OpenCL codes.
Unlike NVIDIA's proprietary OpenCL implementation, pocl supports SPIR
consumption, and so this backend has also been able to run (for example) SYCL
codes using Codeplay's ComputeCpp implementation on NVIDIA GPUs.
Since it uses CUDA under-the-hood, this backend also works with all of the
NVIDIA CUDA profiling and debugging tools, many of which don't work with
NVIDIA's own OpenCL implementation.

Tested platforms
~~~~~~~~~~~~~~~~
The CUDA backend has been tested on Linux (CentOS 7.3) with SM_35, SM_52, and
SM_61 capable NVIDIA GPUs.

The backend is also functional on macOS, with just one additional test failure
compared to Linux (``test_event_cycle``).

Known issues
~~~~~~~~~~~~
The following is a non-comprehensive list of known issues in the CUDA backend:

* image types and samplers are unimplemented
* atomics are unimplemented
* global offsets are unimplemented
* get_work_dim is unimplemented
* printf format support is incomplete

Additionally, there has been little effort to optimize the performance of this
backend so far - the current effort is on implementing remaining functionality.
Once the core functionality is completed, optimization of the code generation
and runtime can begin.

Support
~~~~~~~
For bug reports and questions, please use pocl's `GitHub issue tracker
<https://github.com/pocl/pocl/issues>`_.
Pull requests and other contributions are also very welcome.

This work has primarily been done by James Price from the
`University of Bristol's High Performance Computing Group
<http://uob-hpc.github.io>`_.
