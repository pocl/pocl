==========
TBB device
==========

The TBB device uses the Intel Threading Building Blocks open source (Apache 2.0)
library for scheduling and presents an alternative to the pthread device.

Contrary to the pthread device, the TBB device scheduling characteristics can be
fine tuned with environment variables (see below).

Building pocl with TBB
----------------------

1) Install prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

The Intel Threading Building Blocks library must be available on your system.
The location of ``TBBConfig.cmake`` (shipped with TBB since TBB 2017 U7) is
available via ``TBB_DIR`` or ``CMAKE_PREFIX_PATH`` contains path to TBB root.

2) Build pocl
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

Use the ``POCL_DEVICES`` environment variable to select the TBB device, i. e.
``POCL_DEVICES=tbb``. It is strongly recommended to create only one TBB device
as the TBB device has no subdevice support and always uses all cores.

Optionally, set the ``POCL_TBB_PARTITIONER`` environment variable to one of
``affinity``,``auto``,``simple``,``static`` to select a partitioner. If no
partitioner is selected, the TBB library will select the auto partitioner by
default. More information can be found in the
`related documentation <https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/partitioners.html>`_.

Use optionally ``POCL_TBB_GRAIN_SIZE`` to specify a grain size for all
dimensions. More information can be found in the
`related documentation <https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/Controlling_Chunking.html>`_.
