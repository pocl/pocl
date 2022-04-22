===
HSA
===

Note: pocl's HSA support is currently in experimental stage.

The experimental HSA driver currently only works with generic HSA Agent
support (e.g. for your CPU); implementation exists in the phsa project.

Installing prerequisite software
---------------------------------

1) Install PoCL requirements

  Described in :ref:`_pocl-install`

2) Install an phsa runtime library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Use *phsa* to add generic HSA support on your gcc-supported
  CPU. Its installation instructions are here:

  https://github.com/HSAFoundation/phsa

3) Install the GCC BRIG frontend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  On Ubuntu, it comes packaged::

    sudo apt install gccbrig

4) Build pocl
~~~~~~~~~~~~~

  Using CMake in pocl source directory::

    mkdir build ; cd build
    cmake -DENABLE_HSA=1 -DENABLE_HSAIL=0 -DAMD_HSA=0 ..

  CMake should report "hsa" in list of drivers (OCL_DRIVERS variable).

5) Run tests & play around
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  After building pocl, you can smoke test the HSA driver by executing the HSA
  tests of the pocl testsuite::

    ../tools/scripts/run_hsa_tests

HSA Support notes
------------------

Note that the support is still experimental and very much unfinished. You're
welcome to try it out and report any issues, though.

phsa
~~~~~

`Portable HSA (phsa) <https://github.com/HSAFoundation/phsa>`_ provides similar portable HSA implementation
for CPUs/DSPs and other processors as pocl aims to do for OpenCL. Using phsa one can implement HSA Agent support
for any processor which has a gcc backend with ease.

pocl supports phsa as a backend for its HSA driver, thus any processor utilizing phsa for HSA Agent support
can get OpenCL support via pocl. We used phsa for testing the HSA driver works with other devices and
runtimes than AMD's.


OpenCL 2.0 Atomics and HSA Memory Scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a "memory scope" parameter present in HSA, which applies to atomic memory instructions or
memory fences. Its purpose is to limit the scope of these instructions. However, pocl translates
to HSAIL via LLVM bitcode, and the "atomicrmw" LLVM instruction only takes a memory order parameter, not scope.
For this reason the memory scope in HSAIL is always the widest "system" scope.

Multiple HSA Agent Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While multiple OpenCL device support is not a problem for pocl, the HSA 1.0 specification lacks a "loader/proxy"
feature that OpenCL has in ICD. Thus, support for devices is limited to what the linked HSA runtime supports.

Currently, if one wants to control multiple HSA Agents as multiple pocl OpenCL devices, one needs to implement
a HSA runtime that lists all the Agents to pocl. There is no capability to load multiple HSA runtimes in pocl
as we consider it out of scope and a job for a proxy HSA runtime similar to ICD.

Credits
----------

The current implementation was mainly done by our `Customized Parallel Computing <https://tuni.fi/cpc>`_ group of
Tampere University, Finland with early prototype code contributions from the Programming Language Lab
at National Tsing-Hua University, Hsinchu, Taiwan.

CPC group thanks HSA Foundation and ARTEMIS JU (under grant agreement no 621439, ALMARVI) for funding
this initial pocl HSA driver work. This driver added GPU device support to pocl for the first time, and, on the
other hand, produced an easier path for HSA-supported devices to implement the OpenCL API by utilizing the pocl
code base as a starting point.

In the future we hope to see more effort put in optimizing the results to reach the performance of the
proprietary SDKs on HSA devices.
