.. _pocl-conformance:

=======================
OpenCL conformance
=======================

Conformance related CMake options
---------------------------------

- ``-DENABLE_CONFORMANCE=ON/OFF``
  Defaults to OFF. This option by itself does not guarantee OpenCL-conformant build;
  it merely ensures that a build fails if some CMake options which are known to result
  in non-conformant PoCL build are given. Only applies to CPU and LevelZero drivers.

  Changes when ENABLE_CONFORMANCE is ON, the CPU drivers are built
  with the following changes:

    * read-write images are disabled (some 1D/2D image array tests fail)
    * the list of supported image formats is much smaller
    * SLEEF is always enforced for the builtin library
    * cl_khr_fp16 is disabled
    * cl_khr_subgroup_{ballot,shuffle} are disabled
    * cl_intel_subgroups,cl_intel_required_subgroup_size are disabled

  If ENABLE_CONFORMANCE is OFF, and ENABLE_HOST_CPU_DEVICES is ON,
  the conformance testsuite is disabled in CMake. This is because
  some CTS tests will fail on such build.

Supported & Unsupported optional OpenCL 3.0 features
------------------------------------------------------

This list is only related to CPU devices (cpu & cpu-minimal drivers).
Other drivers (CUDA, TCE etc) only support OpenCL 1.2.
Note that 3.0 support on CPU devices requires LLVM 14 or newer.

Supported 3.0 features:

  * Shared Virtual Memory
  * C11 atomics
  * 3D Image Writes
  * SPIR-V
  * Program Scope Global Variables
  * Subgroups
  * Generic Address Space

Unsupported 3.0 features:

  * Device-side enqueue
  * Pipes
  * Non-Uniform Work Groups
  * Read-Write Images
  * Creating 2D Images from Buffers
  * sRGB & Depth Images
  * Device and Host Timer Synchronization
  * Intermediate Language Programs
  * Program Initialization and Clean-Up Kernels
  * Work Group Collective Functions

.. _running-cts:

How to run the OpenCL 3.0 conformance test suite
------------------------------------------------

You'll need to build PoCL with enabled ICD, and the ICD must be one that supports
OpenCL version 3.0 (for ocl-icd, this is available since version 2.3.0).
This is because while the CTS will run with 1.2 devices, it requires 3.0 headers
and 3.0 ICD to build. You'll also need to enable the suite in the pocl's external test suite set.
This is done by adding ``-DENABLE_TESTSUITES=conformance -DENABLE_CONFORMANCE=ON``
to the cmake command line. After this ``make prepare_examples`` fetches and
prepares the conformance suite for testing. After building pocl with ``make``,
the CTS can be run with ``ctest -L <LABEL>`` where ``<LABEL>`` is a CTest label.

There are two different CTest labels for using CTS, one label covers the full
set tests in CTS, the other contains a much smaller subset of CTS tests. The
smaller is ``conformance_suite_micro_main`` label, which takes approx 10 minutes
on current (PC) hardware. The full sized CTS is available with label
``conformance_suite_full_main``. This can take 10-30 hrs on current
hardware.

If PoCL is compiled with SPIR-V support, two more labels are available, where
``_main`` suffix is replaced by ``_spirv`` (e.g. ``conformance_suite_micro_spirv``)
These labels will run the same tests as the _main variant, but use offline
compilation to produce SPIR-V and use that to create programs,
instead of default creating from OpenCL C source.

Note that running ``ctest -L conformance_suite_micro`` will run *both* variants
(the online and offline compilation) since the -L option takes a regexp.

Additionally, there is a new cmake label, ``conformance_30_only``
to run tests which are only relevant to OpenCL 3.0.

.. _known-issues:

Known issues related to CTS
---------------------------

- With LLVM 15 and 16, when running CTS with the offline compilation mode
  (= via SPIR-V), Clang + SPIR-V translator produce invalid
  SPIR-V for several tests. PoCL bugreport:
  `<https://github.com/pocl/pocl/issues/1232>`_
  Related Khronos issues:
  `<https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2008>`_
  `<https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2024>`_
  `<https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2025>`_


Conformance tests results
-----------------------------------------------------------------------

PoCL has submitted for OpenCL conformance in December 2024, and has been
accepted as conformant product in January 2025. The submission is here:
`<https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_450>`_

For the sumbission, CTS was ran with the following configuration:

  * OS: Ubuntu 24.04.1 LTS
  * Hardware: 12th Gen Intel(R) Core(TM) i9-12900 CPU
  * PoCL Commit: bbe47f3d6
  * Conformant devices: X86-64 CPU with SSE2 or later, AVX, AVX2 or AVX512
