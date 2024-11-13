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
  * Optional: pkg-config
  * Optional: hwloc v1.0 or newer (e.g. `libhwloc-dev`)
  * Optional (but enabled by default): python3 (for support of LLVM bitcode with SPIR target)
  * Optional: llvm-spirv (version-compatible with LLVM) and spirv-tools
    (required for SPIR-V support in CPU / CUDA; Vulkan driver supports SPIR-V through clspv)

For more details, consult the [install guide](http://portablecl.org/docs/html/install.html).

Building PoCL follows the usual CMake build steps. Note however, that PoCL
can be used from the build directory (without installing it system-wide).

## Supported environments

### CI status:

![x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux_gh.yml/badge.svg?event=push&branch=main)
![x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux.yml/badge.svg?event=push&branch=main)
![ARM64](https://github.com/pocl/pocl/actions/workflows/build_arm64.yml/badge.svg?event=push&branch=main)
![CUDA](https://github.com/pocl/pocl/actions/workflows/build_cuda.yml/badge.svg?event=push&branch=main)
![Level Zero](https://github.com/pocl/pocl/actions/workflows/build_level0.yml/badge.svg?event=push&branch=main)
![OpenASIP+Vulkan](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml/badge.svg?event=push&branch=main)
![Remote](https://github.com/pocl/pocl/actions/workflows/build_remote.yml/badge.svg?event=push&branch=main)
![Apple M1](https://github.com/pocl/pocl/actions/workflows/build_macos.yml/badge.svg?event=push&branch=main)
![Windows](https://github.com/pocl/pocl/actions/workflows/build_msvc.yml/badge.svg?event=push&branch=main)

### Support Matrix legend:

:large_orange_diamond: Tested in CI extensively, including OpenCL-CTS tests

:green_circle: : Tested in CI

:yellow_circle: : Should work, but is untested

:x: : Unsupported

### Linux

| CPU device  |     LLVM 14    |     LLVM 15    |     LLVM 16     |     LLVM 17    |     LLVM 18     |
|:------------|:--------------:|:---------------:|:--------------:|:---------------:|:---------------:|
| [x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux_gh.yml) | :green_circle: | :green_circle:  | :green_circle: | :large_orange_diamond: | :large_orange_diamond: |
| [ARM64](https://github.com/pocl/pocl/actions/workflows/build_arm64.yml) | :yellow_circle: | :yellow_circle: |:yellow_circle: | :yellow_circle: | :green_circle:  |
| i686    | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| ARM32   | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| RISC-V  | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| PowerPC | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |

| GPU device  |     LLVM 17    |     LLVM 18     |
|:------------|:--------------:|:---------------:|
| [CUDA SM5.0](https://github.com/pocl/pocl/actions/workflows/build_cuda.yml) | :green_circle: | :green_circle: |
| CUDA SM other than 5.0  | :yellow_circle: | :yellow_circle: |
| [Level Zero](https://github.com/pocl/pocl/actions/workflows/build_level0.yml) | :green_circle: | :green_circle: |
| [Vulkan](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml) | :green_circle: | :x: |

| Special device |    LLVM 17    |     LLVM 18     |
|:---------------|:-------------:|:---------------:|
| [OpenASIP](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml) | :green_circle: | :x:            |
| [Remote](https://github.com/pocl/pocl/actions/workflows/build_remote.yml) | :yellow_circle: | :green_circle:  |
| Remote + RDMA  | :yellow_circle: | :green_circle:  |


### Mac OS X

| CPU device  |     LLVM 16    |     LLVM 17     |
|:------------|:--------------:|:---------------:|
| [Apple M1](https://github.com/pocl/pocl/actions/workflows/build_macos.yml) | :green_circle: | :green_circle:  |

### Windows

| CPU device  |     LLVM 18    |  LLVM 19        |
|:------------|:--------------:|:---------------:|
| MinGW / x86-64   | :yellow_circle: | :yellow_circle:  |
| MSVC / x86-64   | :green_circle: | :green_circle:  |


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
