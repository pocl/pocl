# Portable Computing Language (PoCL)

PoCL is being developed towards an efficient implementation of the OpenCL
standard which can be easily adapted for new targets.

[Official web page](http://portablecl.org)

[Full documentation](http://portablecl.org/docs/html/)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9499/badge)](https://www.bestpractices.dev/projects/9499)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/30739/badge.svg)](https://scan.coverity.com/projects/pocl-pocl)

## Building

This section contains instructions for building PoCL in its default
configuration and a subset of driver backends. You can find the full build
instructions including a list of available options
in the [install guide](http://portablecl.org/docs/html/install.html).

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

For more details, consult the install guide.

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

### GPU support on different architectures

PoCL can be used to provide OpenCL driver on several architectures where the hardware manufacturer does not ship them 
like Nvidia Tegra (ARM) or IBM Power servers. On PPC64le servers, there are specific instructions to handle the build 
of PoCL in [install guide](http://portablecl.org/docs/html/install.html).
See also [PoCL with CUDA driver](#pocl-with-cuda-driver) section for prebuilt
binaries.

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
