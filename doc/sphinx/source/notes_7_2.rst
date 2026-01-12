**************************
Release Notes for PoCL 7.2
**************************

================
Notable bugfixes
================

* Fixed various clLinkProgram issues in the remote driver.
* Fixed remote driver spuriously reconnecting for no apparent reason.

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Implement sub_group_{reduce,scan_exclusive,scan_inclusive}_* and
  sub_group_{all,any,broadcast}. (reduce is only available for PTX6.0+)
* Note that CUDA driver does not support LLVM 21, due to a bug
  in upstream Clang code. Users must use LLVM 18 to 20 with CUDA. For details,
  see https://github.com/llvm/llvm-project/issues/154772

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenASIP (ttasim) driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Upgraded the driver to use OpenASIP v2.2(-pre) which uses LLVM 21.

===================================
Deprecation/feature removal notices
===================================

* Support for LLVM version 17 was removed, versions 18 to 21 are supported
