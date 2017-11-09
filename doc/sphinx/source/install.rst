============
Installation
============

Requirements
------------

In order to build pocl, you need the following support libraries and
tools:

  * Latest released version of LLVM & Clang
  * GNU make
  * libtool dlopen wrapper files (e.g. libltdl3-dev in Debian)
  * pthread (should be installed by default)
  * hwloc v1.0 or newer (e.g. libhwloc-dev)
  * pkg-config
  * cmake

Clang / LLVM Notes
------------------

**IMPORTANT NOTE!** Some platforms (TCE and possibly HSA) require that
you compile & build LLVM with RTTI on. It can be enabled on cmake command
line, as follows:

**Supported versions**

  Note that pocl aims to support **the latest LLVM version** at the time
  of pocl release, **plus the previous** LLVM version. All older LLVM
  versions are supported with "best effort" basis; there might not be
  build bots continuously testing the code base nor anyone fixing their
  possible breakage.

Configure & Build
-----------------

CMake version 2.8.12 or higher is required.

The build+install is the usual CMake way::

  cd <directory-with-pocl-sources>
  mkdir build
  cd build
  cmake [-D<option>=<value> ...] ..
  make && make install

To see the default detected values, run ``cmake ..`` without any options,
it will produce a summary.


CMake: important options & features
-------------------------------------

For multiple-item options, use ";" as separator (you'll have to escape it for bash).

- ``-DWITH_LLVM_CONFIG=<path-to-llvm-config>``
  **IMPORTANT** Path to a llvm-config binary.
  This determines the LLVM installation used by pocl.
  If not specified, pocl will try to find and link against
  llvm-config in PATH env var (usually means your system LLVM).
- ``-DSTATIC_LLVM`` enable this to link LLVM statically into pocl.
  Note that you need LLVM built with static libs. This option might result
  in much longer build/link times and much larger pocl library, but the
  resulting libpocl will not require an LLVM installation to run.
- ``-DENABLE_ICD`` By default pocl's buildsystem will try to find an ICD
  and build pocl as a dynamic library named "libpocl". This option is useful
  if you want to avoid ICD and build pocl directly as libOpenCL library.
  See also :ref:`linking-with-icd`
- ``-DPOCL_INSTALL_<something>_DIR`` The equivalent of ``--bindir``,
  ``--sbindir`` etc fine-tuning of paths for autotools. See the beginning
  of toplevel CMakeLists.txt for all the variables.
- ``-DKERNELLIB_HOST_CPU_VARIANTS`` You can control which CPUs the
  kernel library will be built for. Defaults to "native" which will be
  converted to the build machine's CPU at buildtime. Available CPUs are
  listed by ``llc -mcpu=help``; you can specify multiple CPUs, and pocl will
  look for a kernel library for the runtime-detected CPU.

  For x86(64) there is another possibility, ``distro``, which builds a few
  preselected sse/avx variants covering 99.99% of x86 processors, and pocl
  will use the most appropriate one at runtime, based on detected CPU features.
  With ``distro``, the minimum requirement on CPU is SSE2.

- ``-DENABLE_TESTSUITES`` Which external (source outside pocl) testsuites to enable.
  For the list of testsuites, see examples/CMakeLists.txt or the ``examples``
  directory. Set to ``all`` and pocl will try to autodetect & enable everything
  it can.

  Note that you may build testsuites outside pocl's build tree, and test
  multiple pocl builds with a single testsuite directory. To use this,
  run cmake with ``-DTESTSUITE_BASEDIR=<tests-builddir>`` and ``-DTESTSUITE_SOURCE_BASEDIR=<tests-sourcedir>``.
  The directory structure mirrors that of ``pocl/examples``. So to build e.g. AMD SDK 2.9
  with ``-DTESTSUITE_BASEDIR=/home/pocltest-build -DTESTSUITE_SOURCE_BASEDIR=/home/pocltest-src``,
  place the ``AMD-APP-SDK-v2.9-RC-lnx64.tgz`` file into ``/home/pocltest-src/AMDSDK2.9`` directory.

- ``-DENABLE_CONFORMANCE=ON/OFF``
  Builds Pocl as a fully conformant OpenCL implementation. Defaults to ON.
  See :ref:`pocl-conformance` for details.


LLVM-less build
---------------
 See :ref:`pocl-without-llvm`


Building on Ubuntu 16.04 LTS
----------------------------

The Clang/LLVM 3.8 shipped with Ubuntu 16.04 should work with pocl.
Be sure to install also the 'libclang-3.8-dev' package in addition
to the 'clang-3.8 and llvm-3.8-dev' packages, otherwise cmake will
fail.

Known build-time issues
-----------------------

There are unsolved issues and bugs in pocl. See the bug listing
for a complete listing at https://github.com/pocl/pocl/issues

Known issues not related to pocl are listed below.

- Using Clang compiled with gcc 4.7 causes indeterminism in the
  kernel compilation results. See LLVM bug report:
  http://llvm.org/bugs/show_bug.cgi?id=12945

