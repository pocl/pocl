**************************
Release Notes for PoCL 6.1
**************************

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New experimental support for Defined Built-in Kernels (DBK) has
  been added to the CPU drivers. These DBKs allow for a
  standardized set of built-in kernels with well-defined
  semantics that can be configured during creation of the OpenCL
  program. Currently the following DBKs are implemented: GEMM,
  matrix multiplication, and JPEG en-/de-code. The Extension
  documentation draft can be found on
  `github <https://github.com/KhronosGroup/OpenCL-Docs/pull/1007>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work-group vectorization improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Improve work-group vectorization of kernels which use the global
  id as the iteration variable. Now the loop vectorizer should
  generate wide loads/stores more often than falling back to
  scatter/gather.
* The WG vectorization with the 'loopvec' method is enabled again,
  there was an issue when upgrading to the latest LLVM version.
  Now there's a regression test in place for avoiding accidents
  like this in the futre.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Level Zero driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various improvements were made:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AlmaIF driver (FPGA interfacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

===================================
Notable fixes
===================================

There were a lot of fixed done over the release cycles. Some of the
most notable/user facing ones are listed below:

===================================
Deprecation/feature removal notices
===================================

 * The old "work-item replication" work-group function generation
   method was removed to clean up the kernel compiler. It did not
   anymore have any use cases that could not be covered by fully
   unrolling "loops".
 * The "loops" method does no longer unroll loops explicitly via
   the environment variable POCL_WILOOPS_MAX_UNROLL_COUNT, which
   was removed in this release. Unrolling decisions are delegated
   to standard LLVM optimizations.
