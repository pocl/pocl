
Windows support
-----------------

It is possible to build & use PoCL on Windows using MinGW and
Microsoft Visual Studio (MSVC). Building PoCL on MSVC is recommended
and easiest route. The MinGW route takes more steps in building which
involves and currently requires cross-compilation in a linux machine.

Prerequisites for MSVC Route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Microsoft Visual Studio Community or Professional edition, at least
  version 2019.
* Optional: `Ninja <https://ninja-build.org/>`_.


Building LLVM using MSVC
~~~~~~~~~~~~~~~~~~~~~~~~~~

Open Windows PowerShell and choose a directory as workspace for building LLVM
and PoCL and then::

  git clone https://github.com/llvm/llvm-project.git
  cd llvm-project
  git checkout release/<llvm-version>.x
  cd ..
  cmake -S llvm-project\llvm -B build-llvm -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_INSTALL_PREFIX=<llvm-install-path>\install-llvm
  cmake --build build-llvm --config Release
  cmake --install install-llvm --config Release

Where:

* ``<llvm-version>`` is LLVM major version to be built - e.g. ``19``.

* ``<llvm-install-path>`` is a directory to install the LLVM into and
  used in the PoCL building section ahead.

This should build 64-bit static libraries.

Building PoCL Using MSVC
~~~~~~~~~~~~~~~~~~~~~~~~~~

Source MSVC SDK environment using the command in the PowerShell::

  '& C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64'

If you have professional edition installed instead, replace ``Community`` with
``Professional`` in the above command.

Configure and build PoCL::

  git clone https://github.com/pocl/pocl.git
  cmake -S pocl -B build-pocl -DCMAKE_INSTALL_PREFIX=<pocl-install>\install-pocl -DENABLE_ICD=0 -DENABLE_LLVM=1 -DWITH_LLVM_CONFIG=<llvm-install-path>\bin\llvm-config.exe -DENABLE_LOADABLE_DRIVERS=0 -DSTATIC_LLVM=ON -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DBUILD_SHARED_LIBS=OFF -G "Ninja"
  cmake --build build-pocl
  cmake --install build-pocl

Where ``<llvm-install-path>`` is the directory where the LLVM is
installed in the previous section. This builds PoCL as static library
and building PoCL as dynamic library (``-DBUILD_SHARED_LIBS=ON``) is
not supported yet. ``-G Ninja`` can be replaced with ``-G NMake
Makefiles`` but the building will be very slow.


Running tests from the build directory (MSVC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run PoCL's internal tests delve into the build directory and run::

  $env:POCL_BUILDING = '1'
  ctest -j<N>

Where ``<N>`` is a number of tests to be run in parallel.


Prerequisites for MinGW Route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* a Linux machine with enough memory & space to cross-compile LLVM, and an environment for building containers (docker/podman)
* a Windows machine with UCRT runtime (this comes built-in since Windows 10, available as separate download for earlier versions)

Building LLVM-MinGW
~~~~~~~~~~~~~~~~~~~~~

On the Linux machine, install docker/podman, then execute::

    git clone https://github.com/franz/llvm-mingw.git
    cd llvm-mingw
    ./release.sh

This will produce a file named `llvm-mingw-<TAG>-ucrt-x86_64.zip`, this should contain the required full installation of LLVM 19 + MinGW.
Copy the file to the Windows machine.

Building PoCL using MinGW
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the Windows machine, follow these steps:

* create a directory where the work will happen, e.g. C:\Workspace
* unzip the `llvm-mingw-<TAG>-ucrt-x86_64.zip` in the workspace directory,
  this should create a directory C:\Workspace\llvm-mingw-<TAG>-ucrt-x86_64,
  rename this to `llvm` for convenience
* download the zip version of CMake for Windows (look for 'Windows x64 ZIP' on https://cmake.org/download/),
  unzip it in the C:\Workspace, rename it to `cmake`
* download ninja build tool (look for `ninja-win.zip` on https://github.com/ninja-build/ninja/releases),
  unzip it into the CMake's bin directory
* download the portable zip version of Git SCM from `https://git-scm.com/downloads/win`,
  then unpack it into C:\Workspace, rename to `git`
* optionally, download hwloc release binary from https://www.open-mpi.org/projects/hwloc/,
  unzip and rename to 'hwloc'

From `C:\Workspace\git`, run `git-bash.exe`. In this shell execute the following commands::

    export PATH=/c/Workspace/cmake:$PATH
    export CMAKE_PREFIX_PATH=/c/Workspace/hwloc

    git clone https://github.com/pocl/pocl.git
    cd pocl
    mkdir build
    cd build
    cmake -G Ninja -DENABLE_HWLOC=1 -DENABLE_ICD=0 -DENABLE_LLVM=1 -DSTATIC_LLVM=1 \
       -DWITH_LLVM_CONFIG=/c/Workspace/llvm/bin/llvm-config.exe \
    -DCMAKE_C_COMPILER=/c/Workspace/llvm/bin/clang.exe -DCMAKE_CXX_COMPILER=/c/Workspace/llvm/bin/clang++.exe \
    -DCMAKE_VERBOSE_MAKEFILE=ON -DENABLE_LOADABLE_DRIVERS=0 ..
    ninja -j4

Running tests from the build directory (MinGW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Windows, RPATH is not embedded into binaries. You must set the PATH environment variable to contain
paths of all required DLL libraries; with the packages you've installed in previous step, the DLLs are
placed in the same directory as the binaries. You can use these paths:

* /c/Workspace/pocl/build/lib/CL
* /c/Workspace/hwloc/bin
* /c/Workspace/llvm/bin

Note for debugging: gdb is not installed but lldb.exe is available. The debugged process runs in its
own window and sometimes it exits so quickly there's no time to see the output; in that case, it's
useful to set a breakpoint on exit: "b NtTerminateProcess"
