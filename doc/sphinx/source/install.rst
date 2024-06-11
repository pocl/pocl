.. _pocl-install:

============
Installation
============

Requirements
------------

In order to build pocl, you need the following support libraries and
tools:

  * A supported version of LLVM & Clang (check release notes)
  * development files for LLVM & Clang + their transitive dependencies
    (e.g. libclang-dev, libllvm-dev, zlib1g-dev, libtinfo-dev...)
  * CMake
  * GNU make or ninja
  * pkg-config
  * pthread (should be installed by default)
  * hwloc v1.0 or newer (e.g. libhwloc-dev) - optional
  * python3 (for support of LLVM bitcode with SPIR target; optional but enabled by default)
  * python3, llvm-spirv (version-compatible with LLVM) and spirv-tools (optional;
    required for SPIR-V support in CPU / CUDA; Vulkan driver supports SPIR-V through clspv)

Installing requirements for Ubuntu
----------------------------------

Note: The binary packages from https://apt.llvm.org/ are recommended
(and tested for each release) instead of the binary tar balls or
the packages included in the distribution. The following assumes
apt.llvm.org is added to your apt repos::

    apt install -y python3-dev libpython3-dev build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} llvm-${LLVM_VERSION} make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} llvm-${LLVM_VERSION}-dev

Installing requirements for Arch Linux::

    pacman -S gcc patch hwloc cmake git pkg-config make ninja ocl-icd clang llvm llvm-libs clinfo opencl-headers

Installing requirements for Fedora::

    dnf install gcc gcc-c++ clinfo hwloc-devel hwloc-libs cmake git-core pkgconfig make ninja-build ocl-icd ocl-icd-devel clang clang-devel clang-libs llvm llvm-devel llvm-libs patch redhat-rpm-config findutils

There are also Dockerfiles available for a few most common linux
distributions in ``tools/docker``, looking into them might be helpful.

OpenCL 3.0 support
------------------

If you want PoCL built with ICD and OpenCL 3.0 support at platform level, you will
need sufficiently new ocl-icd (2.3.x). For Ubuntu, it can be installed from
this PPA: https://launchpad.net/~ocl-icd/+archive/ubuntu/ppa

Note: PoCL assumes that the OpenCL development headers and the ICD loader
(if present on your system) are version compatible.

Clang / LLVM Notes
------------------

**IMPORTANT NOTE!** Some targets (TCE and possibly HSA) require that
you compile & build LLVM with RTTI on. It can be enabled on cmake command
line, as follows::

    cmake [other CMake options] -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON <llvm-source-directory>

Supported LLVM versions
~~~~~~~~~~~~~~~~~~~~~~~~

  Note that pocl aims to support **the latest LLVM version** at the time
  of pocl release, **plus the previous** LLVM version. All older LLVM
  versions are supported with "best effort" basis; there might not be
  CI continuously testing the code base nor anyone fixing their
  possible breakage.

Configure & Build
-----------------

CMake version 3.12 or higher is required.

The build+install is the usual CMake way::

  cd <directory-with-pocl-sources>
  mkdir build
  cd build
  cmake [-D<option>=<value> ...] ..
  make && make install

To see the default detected values, run ``cmake ..`` without any options,
it will produce a summary.

CMake variables
~~~~~~~~~~~~~~~~~~~~~~~~

Since pocl includes a compiler for the kernels, it both compiles (producing
code) and is compiled (it consists of code). This distinction typically called
"host" and "target": The host is where pocl is running, the target is
where the OpenCL code will be running. These two systems can be wildly
different.

Host compiler used to compile pocl can be GCC or Clang; the target
compiler is always Clang+LLVM since pocl uses Clang/LLVM internally.
For host compiler, you should use the one which your LLVM was compiled
with (because the LLVM-related parts of pocl take LLVM's CXXFLAGS from
llvm-config and pass them to the host compiler).

CMake host flags
~~~~~~~~~~~~~~~~~~~~~~~~

Compile C:
  CMAKE_C_FLAGS
  CMAKE_C_FLAGS_<build-type>

Compile C++:
  CMAKE_CXX_FLAGS
  CMAKE_CXX_FLAGS_<build-type>

Building kernels and the kernel library, i.e. target flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


All of these empty by default. There are hardcoded defaults which may
be overridden by setting these variables (rarely needed).

Extra parameters to llc
   EXTRA_HOST_LLC_FLAGS

Extra parameters to clang
   EXTRA_HOST_CLANG_FLAGS

Extra parameters to linker (links kernel to shared library
which is then dlopened):

EXTRA_HOST_LD_FLAGS

EXTRA_KERNEL_FLAGS
  is applied to all kernel library compilation commands, IOW it's for
  language-independent options

EXTRA_KERNEL_{C,CL,CXX}_FLAGS
  cmake variables for per-language options for kernel library compilation

.. _pocl-cmake-variables:

CMake: other options & features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that there are a few more packaging-related options described
in ``README.packaging``.

For multiple-item options like KERNELLIB_HOST_CPU_VARIANTS,
use ";" as separator (you'll have to escape it for bash).

- ``-DWITH_LLVM_CONFIG=<path-to-llvm-config>``
  **IMPORTANT** Path to a llvm-config binary.
  This determines the LLVM installation used by pocl.
  If not specified, pocl will try to find and link against
  llvm-config in PATH env var (usually means your system LLVM).

- ``-DSTATIC_LLVM`` pocl uses ``llvm-config --libs`` to get list of LLVM libraries
  it should link to. With this flag enabled, it additionally passes ``--link-static``
  to ``llvm-config``; otherwise it passes ``--link-shared``. Default is OFF (=shared).

- ``-DENABLE_ICD`` By default pocl's buildsystem will try to find an ICD
  and build pocl as a dynamic library named "libpocl". This option is useful
  if you want to avoid ICD and build pocl directly as libOpenCL library.
  See also :ref:`linking-with-icd`

- ``-DPOCL_INSTALL_<something>_DIR`` The equivalent of ``--bindir``,
  ``--sbindir`` etc fine-tuning of paths for autotools. See the beginning
  of toplevel CMakeLists.txt for all the variables.

  Note that if ``CMAKE_INSTALL_PREFIX`` equals ``/usr`` then pocl.icd is
  installed to ``/etc/OpenCL/vendors``, otherwise it's installed to
  ``${CMAKE_INSTALL_PREFIX}/etc/OpenCL/vendors``.

- ``-DLLC_HOST_CPU=<something>``
  Defaults to auto-detection via ``llc``. Run ``llc -mcpu=help``
  for valid values. The CPU type is required to compile
  the "target" (kernel library) part of CPU backend.

  This variable overrides LLVM's autodetected host CPU at configure time.
  Useful when llc fails to detect the CPU (often happens on non-x86
  platforms, or x86 with CPU newer than LLVM).

  Note that when this is set (set by default) and the
  KERNELLIB_HOST_CPU_VARIANTS variable is not ``distro``,
  pocl will first try to find compiled kernel library
  for runtime-detected CPU then fallback to LLC_HOST_CPU.
  This works well if pocl is run where it was built,
  or the actual CPU is in the KERNELLIB_HOST_CPU_VARIANTS list,
  or the actual CPU is >= LLC_HOST_CPU feature-wise;
  otherwise it will likely fail with illegal instruction at runtime.

- ``-DKERNELLIB_HOST_CPU_VARIANTS`` You can control which CPUs the
  "target" part of CPU backend will be built for.
  Unlike LLC_HOST_CPU, this variable is useful if you plan
  to build for multiple CPUs. Defaults to "native" which is
  automagically replaced by LLC_HOST_CPU.
  Available CPUs are listed by ``llc -mcpu=help``. See above for
  runtime CPU detection rules.

  Note that there's another valid value on x86(64) platforms.
  If set to ``distro``, the KERNELLIB_HOST_CPU_VARIANTS variable will be
  set up with a few preselected sse/avx variants covering 99.99% of x86
  processors, and the runtime CPU detection is slightly altered: pocl
  will find the suitable compiled library based on detected CPU features,
  so it cannot fail (at worst it'll degrade to SSE2 library).

- ``-DLLC_TRIPLE=<something>`` Controls what target triple pocl is built for.
  You can set this manually in case the autodetection fails.
  Example value: ``x86_64-pc-linux-gnu``

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

- ``-DENABLE_TESTS=ON/OFF`` enable/disable compilation of internal tests.

- ``-DENABLE_EXAMPLES=ON/OFF`` enable/disable compilation of all examples.
  Disabling this makes ENABLE_TESTSUITES option unavailable.

- ``-DENABLE_POCLCC=ON/OFF`` enable/disable compilation of poclcc.

- ``-DENABLE_CONFORMANCE=ON/OFF``
  Ensures that certain build options which would result in non-conformant pocl
  build stay disabled. Defaults to OFF. Note that this does not quarantee a
  fully conformant build of pocl by itself. See :ref:`pocl-conformance` for details.

- ``-DENABLE_{A,L,T,UB}SAN`` - compiles pocl's host code (and tests
  + examples) with various sanitizers. Using more than one sanitizer at
  a time is untested. Using together with ``-DENABLE_ICD=OFF -DENABLE_LOADABLE_DRIVERS=OFF``
  is highly recommended to avoid issues with loading order of sanitizer libraries.

- ``-DENABLE_LTTNG=ON/OFF`` - compile pocl with LTTng support for tracing. Requires LTTng to be installed
  on the host machine.

- ``-DENABLE_{CUDA,TCE,HSA,VULKAN,LEVEL0}=ON/OFF`` - enable various (non-CPU) backends.
  Usually requires some additional build dependencies; see their documentation.

- ``-DPOCL_DEBUG_MESSAGES=ON`` - when disabled, pocl is compiled without
  debug messages (POCL_DEBUG env var) support.

- ``-DEXAMPLES_USE_GIT_MASTER=ON`` - when enabled, examples (external
  programs in ``examples/`` directory) are built from their git branches
  (if available), as opposed to default: building from release tars.

- ``-DENABLE_POCL_FLOAT_CONVERSION=ON/OFF``
  When enabled, OpenCL printf() call's f/e/g formatters are handled by pocl.
  When disabled (default), these are handled by system C library.

- ``-DINTEL_SDE_AVX512=<PATH>``
  Path to Intel® Software Development Emulator. When this option is given,
  the LLVM host CPU is forcibly set to 'skylake-avx512', and the internal
  tests are run through the Emulator. Mostly useful to test AVX512.

.. _pocl-without-llvm:

LLVM-less build
~~~~~~~~~~~~~~~~~~~~~~~~

You can build a runtime-only pocl to run prebuilt pocl binaries on a device.
To do this

* First, build a pocl with LLVM somewhere.
* on that machine, set up env vars required for your device (if any), then
  run ``bin/poclcc -l``. That should print something like::

    LIST OF DEVICES:
    0:
     Vendor:   AuthenticAMD
       Name:   pthread-AMD A10-7800 Radeon R7, 12 Compute Cores 4C+8G
    Version:   OpenCL 2.0 pocl HSTR: pthread-x86_64-unknown-linux-gnu-bdver3

The string after "HSTR:" is the device build hash.

* now build the LLVM-less pocl. You will need the device build hash from
  previous step:

  ``cmake -DENABLE_LLVM=0 -DHOST_DEVICE_BUILD_HASH=<something> ...``

  This is required because pocl binaries contain a device hash, and the LLVM-less
  pocl needs to know which binaries it can load.

**NOTE**: If you've enabled the :ref:`almaif device <almaif_usage>`
, `HOST_DEVICE_BUILD_HASH` can be set to anything you want. Reason being, fixed function
accelerators don't require compiling OpenCL kernels, therefore, no hash will ever be matched. 

Cross-compile PoCL
------------------
It's now possible to cross-compile pocl on x86-64 to run on ARM/MIPS/etc,
There is a ToolchainExample.cmake file;
copy it under different name, then follow the instructions in the file.

Known build-time issues
------------------------

There are unsolved issues and bugs in pocl. See the bug listing
for a complete listing at https://github.com/pocl/pocl/issues

Building & running in Docker
-----------------------------

Make sure you have enough space (default location is usually ``/var/lib/docker``,
required storage for standard pocl build is about 1.5 GB per container)

Build & start Pocl container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``cd tools/docker``
* pick a Dockerfile from tools/docker, e.g. Fedora/default
* to build PoCL: ``sudo docker build -t TAG -f Fedora/default .``, where
  TAG is a name you choose for the build (must be lowercase)
* to run the tests on the built PoCL: ``sudo docker run -t TAG``
* this will by default use master branch of pocl git; to use a different branch/commit,
  run docker build with ``--build-arg GIT_COMMIT=<branch/commit>``

Dockerfiles
~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that some images (e.g. RHEL and PHSA) may be impossible to build,
due to not having a sufficiently new version of LLVM available.

Dockerfiles are named according to what they build, or the release they're based on:

* `default`: builds pocl, then runs the internal tests from build dir.
   Uses latest release of a distribution, with whatever is the default version of LLVM.
* `distro`: does a distribution-friendly build: enables runtime detection of CPU,
   installs pocl into system path, then runs the internal tests
* `<release>`: same as above, except uses specific release and specific LLVM version
  (the latest available in that release).
* `conformance`: builds & installs Pocl, then runs conformance test suite
  (the shortest version of it)

ARM 32/64bit support
-----------------------------

Status:

PoCL builds (as of Dec 2023) on ODROID XU3 and ODROID C2
and almost all tests pass.

ARM specific build notes:

* DO NOT use Clang/LLVM downloaded directly from llvm.org, they only work
  on the distro where they were compiled. Ubuntu LTS these days ships multiple llvm
  versions even quite recent ones; get Clang+LLVM from your distro's package
  manager or build it yourself.

* LLVM might not recognize your cpu, in which case CMake will complain.
  Run cmake with -DLLC_HOST_CPU=<yourcpu>. "yourcpu" must be something LLVM recognizes,
  usually it's simply "cortex-aXX" like cortex-a15 etc. You can get the full list by
  running `llc -mcpu=help`.

RISC-V support
-----------------------------

The RISC-V support has been tested (as of Dec 2023) on Starfive VisionFive 2 using Ubuntu 23.10 preinstalled image,
with LLVM 17 and GCC 13.2; of the internal tests, 98% tests pass, 4 tests fail out of 253.
In particular, tests using printf with vector arguments are broken ATM. Other boards / CPUs
have not been tested. RISC Vector extension is not supported.

RISC-V specific build notes:

* Avoid older LLVM and GCC versions (like GCC 11 / Clang 14 on the official
  Starfive Debian images) as much as possible. Code generation is much
  better with recent versions, and your experience will generally better

* LLVM might not recognize your CPU, in which case CMake will complain.
  Run cmake with -DLLC_HOST_CPU=<yourcpu>. "yourcpu" must be something LLVM recognizes;
  you can get the full list by running `llc -mcpu=help`.

* on RISC-V, PoCL additionally needs to pass a target ABI flag to the compiler. There is
  some autodetection in PoCL but right now it's limited, and Clang unfortunately does not
  always get the defaults correctly. If you get errors similar to:

      "can't link double-float modules with soft-float modules"

  from linker, then most likely PoCL used the incorrect ABI. You can explicitly
  specify the ABI to use with the HOST_CPU_TARGET_ABI CMake option.
