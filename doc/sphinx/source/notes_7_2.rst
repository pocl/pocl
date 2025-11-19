**************************
Release Notes for PoCL 7.2
**************************

===================================
Experimental and work-in-progress
===================================

* Added experimental MLIR-based OpenCL C kernel compilation flow for CPUs.
  The flow supports both Polygeist and ClangIR front-ends.
  Support for basic features such as local memory and some barrier scenarios is included,
  but the majority of built-in functions not yet supported.
  Contributions are welcome to increase the test coverage.
  Currently at about 36\% of PoCL tests with the Polygeist front-end.

