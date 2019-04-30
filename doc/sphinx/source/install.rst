.. _pocl-install:

============
Installation
============

Requirements
------------

In order to build pocl, you need the following support libraries and
tools:

  * Latest released version of LLVM & Clang
  * development files for LLVM & Clang + their transitive dependencies
    (e.g. libclang-dev, libllvm-dev, zlib1g-dev, libtinfo-dev...)
  * GNU make or ninja
  * libtool dlopen wrapper files (e.g. libltdl3-dev in Debian)
  * pthread (should be installed by default)
  * hwloc v1.0 or newer (e.g. libhwloc-dev)
  * pkg-config
  * cmake


There are Dockerfiles available for a few most common linux
distributions in ``tools/docker``, looking into them might be helpful.

Clang / LLVM Notes
------------------

**IMPORTANT NOTE!** Some targets (TCE and possibly HSA) require that
you compile & build LLVM with RTTI on. It can be enabled on cmake command
line, as follows:

    cmake .... -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON ....

Supported LLVM versions
~~~~~~~~~~~~~~~~~~~~~~~~

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

- ``-DSTATIC_LLVM`` this option is deprecated and currently has no effect. See
  the SINGLE_LLVM_LIB option.

- ``-DSINGLE_LLVM_LIB`` when this option is enabled (default), pocl tries to
  link to a single big LLVM library (libLLVM-<VERSION>.suffix). If this fails,
  it fallbacks to linking LLVM libraries provided by ``llvm-config --libfiles``.

- ``-DENABLE_ICD`` By default pocl's buildsystem will try to find an ICD
  and build pocl as a dynamic library named "libpocl". This option is useful
  if you want to avoid ICD and build pocl directly as libOpenCL library.
  See also :ref:`linking-with-icd`

- ``-DENABLE_FP64`` - for ARM platform only. If your CPU doesn't support any
  doubles (VFP is enough), disable this. Defaults to OFF when LLVM is older
  than 4.0, otherwise defaults to ON.

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
  Ensures that certain build options which would result in non-conformant pocl
  build stay disabled. Defaults to ON. Note that this does not quarantee a
  fully conformant build of pocl by itself. See :ref:`pocl-conformance` for details.

- ``-DENABLE_{A,L,T,UB}SAN`` - compiles pocl's host code (and tests
  + examples) with various sanitizers. Using more than one sanitizer at
  a time is untested. Using together with ``-DENABLE_ICD=OFF`` is highly
  recommended to avoid issues with loading order of sanitizer libraries.

- ``-DENABLE_{CUDA,TCE,HSA}=ON/OFF`` - enable various (non-CPU) backends.
  Usually requires some extra setup; see their documentation.

- ``-DPOCL_DEBUG_MESSAGES=ON`` - when disabled, pocl is compiled without
  debug messages (POCL_DEBUG env var) support.

- ``-DEXAMPLES_USE_GIT_MASTER=ON`` - when enabled, examples (external
  programs in ``examples/`` directory) are built from their git branches
  (if available), as opposed to default: building from release tars.

- ``-DENABLE_POCL_FLOAT_CONVERSION=ON/OFF``
  When enabled, OpenCL printf() call's f/e/g formatters are handled by pocl.
  When disabled (default), these are handled by system C library. Can only
  be enabled when Clang's compiler-rt library is present.

- ``-DINTEL_SDE_AVX512=<PATH>``
  Path to Intel® Software Development Emulator. When this option is given,
  the LLVM host CPU is forcibly set to 'skylake-avx512', and the internal
  tests are run through the Emulator. Mostly useful to test AVX512.


LLVM-less build
~~~~~~~~~~~~~~~~~~~~~~~~
 See :ref:`pocl-without-llvm`


Known build-time issues
~~~~~~~~~~~~~~~~~~~~~~~~

There are unsolved issues and bugs in pocl. See the bug listing
for a complete listing at https://github.com/pocl/pocl/issues

Known issues not related to pocl are listed below.

- Using Clang compiled with gcc 4.7 causes indeterminism in the
  kernel compilation results. See LLVM bug report:
  http://llvm.org/bugs/show_bug.cgi?id=12945

