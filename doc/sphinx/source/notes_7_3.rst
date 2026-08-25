**************************
Release Notes for PoCL 7.3
**************************

===========================
Release highlights
===========================

================
CMake changes
================

* Added an `ENABLE_CUDA_IMAGES` option. Note that image support in the CUDA
  driver is still experimental and very incomplete.

==========================
Runtime fixes & features
==========================

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Initial bits for images support. Currently only the `IMAGE1D_BUFFER` image
  type is supported.
