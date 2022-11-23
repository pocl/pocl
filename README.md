# Portable Computing Language (PoCL)

PoCL is being developed towards an efficient implementation of the OpenCL
standard which can be easily adapted for new targets.

[Official web page](http://portablecl.org)

[Full documentation](http://portablecl.org/docs/html/)

## Building

This section contains instructions for building PoCL in its default
configuration and a subset of driver backends. You can find the full build
instructions including a list of available options
in the [user guide](http://portablecl.org/docs/html/install.html).

### Requirements

In order to build PoCL, you need the following support libraries and
tools:

  * Latest released version of LLVM & Clang
  * development files for LLVM & Clang + their transitive dependencies
    (e.g. `libclang-dev`, `libclang-cpp-dev`, `libllvm-dev`, `zlib1g-dev`,
    `libtinfo-dev`...)
  * CMake
  * GNU make or ninja
  * pkg-config
  * pthread (should be installed by default)
  * hwloc v1.0 or newer (e.g. `libhwloc-dev`) - optional
  * python3 (for support of LLVM bitcode with SPIR target; optional
    but enabled by default)
  * llvm-spirv (version-compatible with LLVM) and spirv-tools
    (optional; required for SPIR-V support in CPU / CUDA; Vulkan driver
    supports SPIR-V through clspv)

On Ubuntu or Debian based distros you can install the relevant packages with
```
export LLVM_VERSION=<major LLVM version, latest and previous are supported>
apt install -y python3-dev libpython3-dev build-essential ocl-icd-libopencl1 \
    cmake git pkg-config libclang-${LLVM_VERSION}-dev clang \
    llvm-${LLVM_VERSION} make ninja-build ocl-icd-libopencl1 ocl-icd-dev \
    ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils \
    libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev
```

If your distro does not package the version of LLVM you wish to build against
you might want to set up the
[upstream LLVM package repository](https://apt.llvm.org/).

Building on ARM platforms is possible but lacks a maintainer and there are
[some gotchas](./README.ARM).

If you are a distro maintainer, check [README.packaging](./README.packaging) for
recommendations on build settings for packaged builds.

### OpenCL 3.0 support

If you want PoCL built with ICD and OpenCL 3.0 support at platform level,
you will need sufficiently new ocl-icd (2.3.x). For Ubuntu, it can be installed'
from this PPA: https://launchpad.net/~ocl-icd/+archive/ubuntu/ppa
Additionally, if you want the CPU device to report as 3.0 OpenCL
you will need LLVM 14 or newer.

### Windows

Windows support has been unmaintained for a long time and building on Windows
may or may not work. There are old instructions for building with Visual Studio
in [README.Windows](./README.Windows) but with the builtin CMake support of more
recent Visual Studio versions (2019+) it might be enough to install the
dependencies (e.g. with `winget`) and simply open the main `CMakeLists.txt` file
in Visual Studio and let it work its magic.

Contributions for improving compatibility with Windows and more detailed and up
to date build steps are welcome!