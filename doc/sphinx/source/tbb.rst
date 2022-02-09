==========
TBB device
==========

The TBB device uses the Intel Threading Building Blocks open source (Apache 2.0)
library for work-group and kernel-level task scheduling.

The TBB device scheduling characteristics can be fine tuned with environment
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

When building the tbb device, it will be set as the default device for PoCL. It
is strongly recommended to **NOT** create more TBB devices as the TBB device
always uses all cores and has no subdevice support.

Optionally, set the ``POCL_TBB_PARTITIONER`` environment variable to one of
``affinity``,``auto``,``simple``,``static`` to select a partitioner. If no
partitioner is selected, the TBB library will select the auto partitioner by
default. More information can be found in the
`related documentation <https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/partitioners.html>`_.

Use optionally ``POCL_TBB_GRAIN_SIZE`` to specify a grain size for all
dimensions. More information can be found in the
`related documentation <https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/Controlling_Chunking.html>`_.
