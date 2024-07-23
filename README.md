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
  * CMake 3.9 or newer
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
```bash
export LLVM_VERSION=<major LLVM version>
apt install -y python3-dev libpython3-dev build-essential ocl-icd-libopencl1 \
    cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} \
    llvm-${LLVM_VERSION} make ninja-build ocl-icd-libopencl1 ocl-icd-dev \
    ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils \
    libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev
```

For LLVM 18 and above also `libpolly-${LLVM_VERSION}-dev libzstd-dev` are
needed for static LLVM linkage.

If your distro does not package the version of LLVM you wish to build against
you might want to set up the
[upstream LLVM package repository](https://apt.llvm.org/).

### Configure & Build

Building PoCL follows the usual CMake workflow, i.e.:
```bash
cd <directory-with-pocl-sources>
mkdir build
cd build
cmake ..
make
# and optionally
make install
```

### Supported LLVM Versions

PoCL aims to support **the latest LLVM version** at the time of PoCL release, **plus the previous** LLVM version. All older LLVM versions are supported on a
"best effort" basis; there might not be build bots continuously testing the code
base nor anyone fixing their possible breakage.

### OpenCL 3.0 support

If you want PoCL built with ICD and OpenCL 3.0 support at platform level,
you will need sufficiently new ocl-icd (2.3.x). For Ubuntu, it can be installed'
from this PPA: https://launchpad.net/~ocl-icd/+archive/ubuntu/ppa
Additionally, if you want the CPU device to report as 3.0 OpenCL
you will need LLVM 14 or newer.

### GPU support on different architectures

PoCL can be used to provide OpenCL driver on several architectures where the hardware manufacturer does not ship them 
like Nvidia Tegra (ARM) or IBM Power servers. On PPC64le servers, there are specific instructions to handle the build 
of PoCL in [README.PPC64le](./README.PPC64le).
See also [PoCL with CUDA driver](#pocl-with-cuda-driver) section for prebuilt
binaries.

### Windows

Windows support has been unmaintained for a long time and building on Windows
may or may not work. There are old instructions for building with Visual Studio
in [README.Windows](./README.Windows) but with the builtin CMake support of more
recent Visual Studio versions (2019+) it might be enough to install the
dependencies (e.g. with `winget`) and simply open the main `CMakeLists.txt` file
in Visual Studio and let it work its magic.

Contributions for improving compatibility with Windows and more detailed and up
to date build steps are welcome!

### Notes

Building on ARM platforms is possible but lacks a maintainer and there are
[some gotchas](./README.ARM).

If you are a distro maintainer, check [README.packaging](./README.packaging) for
recommendations on build settings for packaged builds.

## Binary packages

### Linux distros

PoCL with CPU device support can be found on many linux distribution managers.
See [![latest packaged version(s)](https://repology.org/badge/latest-versions/pocl.svg)](https://repology.org/project/pocl/versions)

### PoCL with CUDA driver

PoCL with CUDA driver support for Linux `x86_64`, `aarch64` and `ppc64le`
can be found on conda-forge distribution and can be installed with

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh   # install mambaforge

To install pocl with cuda driver

    mamba install pocl-cuda

To install all drivers

    mamba install pocl

### macOS

#### Homebrew

PoCL with CPU driver support Intel and Apple Silicon chips can be
found on homebrew and can be installed with

    brew install pocl

Note that this installs an ICD loader from KhronoGroup and the builtin
OpenCL implementation will be invisible when your application is linked
to this loader.

#### Conda

PoCL with CPU driver support Intel and Apple Silicon chips
can be found on conda-forge distribution and can be installed with

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh

To install the CPU driver

    mamba install pocl

Note that this installs an ICD loader from KhronoGroup and the builtin
OpenCL implementation will be invisible when your application is linked
to this loader. To make both pocl and the builtin OpenCL implementaiton
visible, do

    mamba install pocl ocl_icd_wrapper_apple

## License

PoCL is distributed under the terms of the MIT license. Contributions are expected
to be made with the same terms.

