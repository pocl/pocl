.. _mlir-label:

=========================
Experimental MLIR Support
=========================

There is highly experimental support for using the LLVM's MLIR project to generate work-group functions.
The motivation for this work is to use MLIR's parallel dialects,
which enables the PoCL kernel compiler to plug into existing MLIR optimization and backend frameworks.
Another motivation is the availability of MLIR-based high-level synthesis tools such as
`CIRCT <https://github.com/llvm/circt>`_ or `HIDA <https://github.com/UIUC-ChenLab/ScaleHLS-HIDA>`_.
The longer term goal is then to use OpenCL via PoCL as an ASIC/FPGA HLS frontend.

The MLIR-based flow uses a different front-end compiler than PoCL's LLVM-based flow.
There are two supported front-end options: Polygeist and ClangIR
(with experimental *ThroughMLIR*-compilation).
The front-end produces SPMD programs that use upstream MLIR dialects such as affine, arith, and scf.

The MLIR-based compiler links in built-in functions pre-compiled to MLIR IR from `lib/kernel/mlir`,
then it performs a de-SPMD conversion using a pass in `lib/mliropencl/lib/Transforms/Workgroup.cpp`.

Then, the MLIR IR is lowered to a specific device.
For CPU devices it is lowered to an LLVM IR form that is compatible with the PoCL's typical LLVM-based flow.
After that, it is executed on the CPU device using the same codepath as with the LLVM-based flow.

Building
^^^^^^^^

For the MLIR compiler front-end, there are two options: Polygeist and ClangIR.
In order to use the ClangIR front-end, very recent
`upstream ClangIR <https://github.com/llvm/clangir/tree/d4ebb05f347d8d9d62968676d5b2bbc1338de499>`_
is required.
Newer revisions may also work, but are untested.

When building ClangIR, set::

  -DLLVM_ENABLE_PROJECTS="mlir;clang"
  -DCLANG_ENABLE_CIR=ON
  -DLLVM_LINK_LLVM_DYLIB=ON

When building PoCL, set::

  -DENABLE_CLANGIR=ON -DMLIR_DIR=/path/to/mlir-install/lib/cmake/mlir/

Polygeist currently works with more tests.
However, since development activity on Polygeist is very low,
ClangIR might overtake it in the future,
even though it’s currently failing many cases that work with Polygeist.

The current version supports only `the specific Polygeist fork <https://github.com/cpc/polygeist>`_.
You can enable Polygeist when building PoCL with::

  -DPOLYGEIST_BINDIR=/path/to/cgeist

Usage
^^^^^

The tests that are known to pass with Polygeist+MLIR are labeled with *mlir*,
and can be run with:

  ctest -L mlir

To test MLIR passes separately, you can run LIT-tests.
You need to set -DLLVM_EXTERNAL_LIT= in CMake to configure the correct LIT binary.
LIT-tests use FileCheck, so when ClangIR is built, the -DLLVM_INSTALL_UTILS should be ON
in order for PoCL to find the FileCheck binary.
If LLVM_EXTERNAL_LIT is not set and FileCheck is not found,
the check-pocl-mlir-opt CMake target is not created,
but the MLIR-compiler will still work.
To run the tests:

  make check-pocl-mlir-opt

Supported Features
^^^^^^^^^^^^^^^^^^

In its current state, the MLIR-based compiler flow supports only a very limited set of OpenCL features.
PolybenchGPU benchmarks work with both ClangIR and Polygeist front-ends.
The benchmark suite can be enabled as an external testsuite with the -DENABLE_TESTSUITES=polybenchGPU CMake parameter.

Non-exhaustive list of unimplemented features:

* Barriers
* OpenCL vector datatypes
* Most built-in functions
