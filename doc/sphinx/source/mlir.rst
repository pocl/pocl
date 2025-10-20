.. _mlir-label:

=========================
Experimental MLIR Support
=========================

There is highly experimental support for using the LLVM's MLIR project to generate work-group functions.
The motivation for this work is to use MLIR's parallel dialects,
which enables the PoCL kernel compiler to plug into existing MLIR optimization and backend frameworks.
Another motivation is the availability of MLIR-based high-level synthesis tools such as
`CIRCT <https://github.com/llvm/circt>`_ or `HIDA <https://github.com/UIUC-ChenLab/ScaleHLS-HIDA>`_.
which opens the door to using OpenCL via PoCL as an ASIC/FPGA HLS frontend.

The MLIR-based flow uses a different front-end compiler than PoCL's LLVM-based flow.
There are two supported front-end options: Polygeist and ClangIR
(with experimental *ThroughMLIR*-compilation).
The front-end produces SPMD programs that use upstream MLIR dialects such as affine, arith, and scf.

The MLIR-based compiler links in built-in functions pre-compiled to MLIR IR from `lib/kernel/mlir`,
then it performs a de-SPMD conversion using a pass in `lib/mliropencl/lib/Transforms/Workgroup.cpp`.
After that, any barrier operations are eliminated using
`a pass ported from Polygeist <https://github.com/cpc/clangir/blob/main/mlir/lib/Polygeist/Transforms/DistributeBarriers.cpp>`_

Then, the MLIR IR is lowered to a specific device.
For CPU devices it is lowered to an LLVM IR form that is compatible with the PoCL's typical LLVM-based flow.
After that, it is executed on the CPU device using the same codepath as with the LLVM-based flow.

Building
^^^^^^^^

The current version requires `the specific ClangIR fork <https://github.com/cpc/clangir>`_.

When building ClangIR, set::

  -DLLVM_ENABLE_PROJECTS="mlir;clang"
  -DCLANG_ENABLE_CIR=ON
  -DLLVM_ENABLE_RTTI=ON
  -DLLVM_LINK_LLVM_DYLIB=ON

When building PoCL, set::

  -DENABLE_CLANGIR=ON -DMLIR_DIR=/path/to/mlir-install/lib/cmake/mlir/

For the front-end, there are two options: Polygeist and ClangIR.
Polygeist currently works with more tests.
However, since development activity on Polygeist is quite low,
ClangIR might overtake it in the future,
even though it’s currently failing many cases that work with Polygeist.

The current version supports only `the specific Polygeist fork <https://github.com/cpc/polygeist>`_.
You can enable Polygeist when building PoCL with::

  -DPOLYGEIST_BINDIR=/path/to/cgeist

Usage
^^^^^

To test MLIR passes separately, you can run LIT-tests.
You may need to set -DLLVM_EXTERNAL_LIT= in CMake to configure the correct LIT binary,

  make check-pocl-opt

The tests that are known to pass with Polygeist+MLIR are labeled with *mlir*,
and can be run with:

  ctest -L mlir

Supported Features
^^^^^^^^^^^^^^^^^^

The MLIR-based compiler flow supports many OpenCL features such as local memory and various barrier scenarios
(including some nested ifs and loops).
However, there are some known miscompilations with barriers inside loops.
The easiest way to find these is to attempt to run the PoCL's CPU tests with *-R barrier*.

PolybenchGPU benchmarks work with both ClangIR and Polygeist front-ends.
The benchmark suite can be enabled as an external testsuite with the -DENABLE_TESTSUITES=polybenchGPU CMake parameter.

Non-exhaustive list of unimplemented features:

* OpenCL vector datatypes
* Most built-in functions
