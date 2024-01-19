**********************
CPU device drivers
**********************

========================
'cpu' driver
========================

This is the default CPU driver, using the pthread library for multithreaded execution.
This driver is the most mature and passes almost entirely the conformance test suite.
Building of this driver is enabled by default, but can be disabled by -DENABLE_HOST_CPU_DEVICES=0.

========================
'cpu-minimal' driver
========================

A minimalistic example CPU device driver for executing kernels on the host CPU. Does not
support multithreading.

========================
'cpu-tbb' driver
========================

This driver uses the Intel Threading Building Blocks (currently named oneTBB) open source library
for work-group and kernel-level task scheduling.

The scheduling characteristics can be fine tuned with environment
variables (see below) to achieve a higher performance.

Building PoCL with TBB
----------------------

1) Install prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

The Intel Threading Building Blocks library must be available on your system.
The location of ``TBBConfig.cmake`` (shipped with TBB since TBB 2017 U7) is
available via ``TBB_DIR`` or ``CMAKE_PREFIX_PATH`` contains path to TBB root.

2) Build PoCL
~~~~~~~~~~~~~

To enable the TBB device, add ``-DENABLE_TBB_DEVICE=1`` to your CMake
configuration command line.

If CMake has trouble locating the TBB library, try specifying the to path to
``TBBConfig.cmake`` by passing ``-DTBB_DIR=<path-to-TBBConfig.cmake>`` to CMake.
Examples::

  -DTBB_DIR=/home/username/intel/tbb_2021/lib/cmake/tbb
  -DTBB_DIR=/home/username/intel/tbb_2020/cmake

3) Configuration
~~~~~~~~~~~~~~~~

When building the tbb device, it will be set as the default device for PoCL instead of pthread driver.
It is strongly recommended to **NOT** create more TBB devices as the TBB device
always uses all cores and has no subdevice support.

The TBB driver can be tuned with at runtime with environment variables, see :ref:`pocl-env-variables`.
