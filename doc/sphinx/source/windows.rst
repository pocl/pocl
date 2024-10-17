
Windows support
-----------------


It is possible to build & use PoCL on Windows using MinGW (MSVC port is in progress). However, the setup requires a number of steps.

Prerequisites
~~~~~~~~~~~~~~~

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

Building PoCL
~~~~~~~~~~~~~~~~~

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
    -DCMAKE_C_COMPILER=/c/Workspace/llvm/bin/clang.exe -DCMAKE_CXX_COMPILER=/c/Workspace/llvm/bin/clang++.exe \
    -DCMAKE_VERBOSE_MAKEFILE=ON -DENABLE_LOADABLE_DRIVERS=0 -DENABLE_PLATFORM_CPP=1 -DVISIBILITY_HIDDEN=0 ..
    ninja -j4

Running tests from the build directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Windows, RPATH is not embedded into binaries. You must set the PATH environment variable to contain
paths of all required DLL libraries; with the packages you've installed in previous step, the DLLs are
placed in the same directory as the binaries. You can use these paths:

* /c/Workspace/pocl/build/lib/CL
* /c/Workspace/hwloc/bin
* /c/Workspace/llvm/bin

Note for debugging: gdb is not installed but lldb.exe is available. The debugged process runs in its
own window and sometimes it exits so quickly there's no time to see the output; in that case, it's
useful to set a breakpoint on exit: "b NtTerminateProcess"
