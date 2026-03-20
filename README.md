# Portable Computing Language (PoCL)

PoCL is a conformant implementation (for [CPU](https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_450)
and [Level Zero GPU](https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_453) targets)
of the OpenCL 3.0 standard which can be easily adapted for new targets.

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
  * CMake 3.15 or newer
  * GNU make or ninja
  * Optional: pkg-config
  * Optional: hwloc v1.0 or newer (e.g. `libhwloc-dev`)
  * Optional (but enabled by default): python3 (for support of LLVM bitcode with SPIR target)
  * Optional: llvm-spirv (version-compatible with LLVM) and spirv-tools
    (required for SPIR-V support in CPU / CUDA; Vulkan driver supports SPIR-V through clspv)

For more details, consult the [install guide](http://portablecl.org/docs/html/install.html).

Building PoCL follows the usual CMake build steps. Note however, that PoCL
can be used from the build directory (without installing it system-wide).

## Device drivers

PoCL supports several backend drivers, with different levels of maturity
in terms of received testing, reliability and available features.
CPU/x86_64 is continuously tested to pass CTS, and is also able to pass >99%
of all CTS tests when built with Thread or Address sanitizers.
CPU driver is also tested on RISCV and ARM64;
CPU driver on ARM32, i386, PPC, S390x is not tested or supported. We won't
prevent building on these architectures, but we don't actively support them ATM.


|     Driver  |   Maturity      |  OpenCL version  | input SPIR-V   |
|:------------|:---------------:|:----------------:|:--------------:|
| CPU/x86_64  |   very high     |    3.0           |  1.4           |
| CPU/ARM64   |   high          |    3.0           |  1.4           |
| CPU/RISCV   |   high          |    3.0           |  1.4           |
| LevelZero   |   high          |    3.0           |  as LZ runtime |
| CUDA        |   low           |    3.0           |  1.2           |
| OpenASIP    |   low           |    1.2           |  none          |
| Vulkan      |   low           |    3.0           |  ExecModel=Shader only |

## Supported CI environments

### CI status:

![x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux_gh.yml/badge.svg?event=push&branch=main)
![x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux.yml/badge.svg?event=push&branch=main)
![ARM64](https://github.com/pocl/pocl/actions/workflows/build_arm64.yml/badge.svg?event=push&branch=main)
![CUDA](https://github.com/pocl/pocl/actions/workflows/build_cuda.yml/badge.svg?event=push&branch=main)
![Level Zero](https://github.com/pocl/pocl/actions/workflows/build_level0.yml/badge.svg?event=push&branch=main)
![OpenASIP+Vulkan](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml/badge.svg?event=push&branch=main)
![Remote](https://github.com/pocl/pocl/actions/workflows/build_remote.yml/badge.svg?event=push&branch=main)
![Apple Silicon](https://github.com/pocl/pocl/actions/workflows/build_macos.yml/badge.svg?event=push&branch=main)
![Windows](https://github.com/pocl/pocl/actions/workflows/build_msvc.yml/badge.svg?event=push&branch=main)

### Support Matrix legend:

:large_blue_diamond: Achieved status of OpenCL conformant implementation

:large_orange_diamond: Tested in CI extensively, including OpenCL-CTS tests

:green_circle: : Tested in CI

:yellow_circle: : Should work, but is untested

:red_circle: : Unsupported

### Linux

| CPU device  |     LLVM 18     |      LLVM 19     |     LLVM 20     |     LLVM 21     |     LLVM 22     |
|:------------|:---------------:|:----------------:|:---------------:|:---------------:|:---------------:|
| [x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux.yml) | :large_blue_diamond: | :green_circle: | :green_circle: | :large_orange_diamond: | :large_orange_diamond: |
| [ARM64](https://github.com/pocl/pocl/actions/workflows/build_arm64.yml)  | :yellow_circle: | :yellow_circle: |  :yellow_circle: | :yellow_circle: | :green_circle: |
| i686    | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| ARM32   | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| RISC-V  | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :green_circle:  |
| PowerPC | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |

| GPU device  |     LLVM 18     |      LLVM 19     |     LLVM 20     |     LLVM 21     |     LLVM 22     |
|:------------|:---------------:|:----------------:|:---------------:|:---------------:|:---------------:|
| [CUDA SM5.0](https://github.com/pocl/pocl/actions/workflows/build_cuda.yml) | :yellow_circle: | :yellow_circle: | :green_circle: | :red_circle: | :green_circle: |
| CUDA SM other than 5.0                                                      | :yellow_circle: | :yellow_circle: | :yellow_circle: | :red_circle: | :yellow_circle: |
| [Level Zero](https://github.com/pocl/pocl/actions/workflows/build_level0.yml) | :yellow_circle: | :yellow_circle: | :green_circle: | :large_orange_diamond: | :green_circle: |
| [Vulkan](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml) | :green_circle: | :red_circle: | :red_circle: | :red_circle: | :red_circle: |

Note: CUDA with LLVM 21 is broken due to a bug in Clang (https://github.com/llvm/llvm-project/issues/154772).

| Special device |     LLVM 18     |      LLVM 19     |     LLVM 20     |     LLVM 21     |      LLVM 22     |
|:---------------|:---------------:|:----------------:|:---------------:|:---------------:|:---------------:|
| [OpenASIP](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml) | :red_circle: | :red_circle: |  :red_circle: | :green_circle: |  :red_circle: |
| [Remote](https://github.com/pocl/pocl/actions/workflows/build_remote.yml) | :green_circle: | :green_circle:  | :green_circle: | :green_circle: | :yellow_circle: |


### Mac OS X

| CPU device  |     LLVM 18     |      LLVM 19     |     LLVM 20     |      LLVM 21     |      LLVM 22     |
|:------------|:---------------:|:----------------:|:---------------:|:----------------:|:----------------:|
| [Apple Silicon](https://github.com/pocl/pocl/actions/workflows/build_macos.yml) | :yellow_circle: | :yellow_circle: | :green_circle: | :green_circle: | :yellow_circle: |
| [Intel CPU](https://github.com/pocl/pocl/actions/workflows/build_macos.yml)     | :yellow_circle: | :red_circle: | :red_circle: | :red_circle: | :red_circle: |

### Windows

| CPU device  |     LLVM 18    |  LLVM 19        |     LLVM 20     |     LLVM 21     |      LLVM 22     |
|:------------|:--------------:|:---------------:|:---------------:|:---------------:|:----------------:|
| [MinGW](https://github.com/pocl/pocl/actions/workflows/build_mingw.yml) / x86-64  | :yellow_circle: | :green_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| [MSVC](https://github.com/pocl/pocl/actions/workflows/build_msvc.yml) / x86-64    | :yellow_circle: | :green_circle: | :green_circle:  | :yellow_circle: | :yellow_circle: |

## Supported OpenCL extensions & OpenCL C features

### Support legend:

:green_circle: : Supported with all hardware & LLVM versions

:yellow_circle: : Partially supported, see notes

:red_circle: : Unsupported

empty cell : Unknown status

### OpenCL extensions

Some extensions are available at Platform level:
* `cl_khr_icd` - if compiled with ENABLE_ICD=1
* `cl_khr_create_command_queue`
* `cl_pocl_content_size`
* `cl_ext_buffer_device_address`


| Extension | CPU device | Level Zero | CUDA | OpenASIP | Remote |
|:----------|:----------:|:----------:|:----:|:--------:|:------:|
| cl_khr_byte_addressable_store | :green_circle: | :green_circle: | :green_circle: | | |
| cl_khr_global_int32_base_atomics | :green_circle: | :green_circle: | :green_circle: | | |
| cl_khr_global_int32_extended_atomics | :green_circle: | :green_circle: | :green_circle: | | |
| cl_khr_local_int32_base_atomics | :green_circle: | :green_circle: | :green_circle: | | |
| cl_khr_local_int32_extended_atomics | :green_circle: | :green_circle: | :green_circle: | | |
| cl_khr_int64_base_atomics | :green_circle: | :yellow_circle: :two: :one: | :green_circle: | | |
| cl_khr_int64_extended_atomics | :green_circle: | :yellow_circle: :two: :one: | :green_circle: | | |
| cl_khr_suggested_local_work_size | :green_circle: | | | | |
| cl_khr_device_uuid | :green_circle: | :green_circle: | | | |
| cl_khr_pci_bus_info | :red_circle: | :yellow_circle: :one: | | | |
| cl_intel_device_attribute_query | :red_circle: | :yellow_circle: :one: | | | |
| cl_khr_3d_image_writes | :green_circle: | :yellow_circle: :two: | | | |
| cl_khr_depth_images | :red_circle: | :yellow_circle: :two: | | | |
| cl_khr_integer_dot_product | :green_circle: | :yellow_circle: :one: | | | |
| cl_ext_float_atomics | :green_circle: | :yellow_circle: :one: | :green_circle: | | |
| cl_intel_unified_shared_memory | :green_circle: | :yellow_circle: :one: | | | |
| cl_ext_buffer_device_address | :green_circle: | :yellow_circle: :one: | | | |
| cl_khr_extended_bit_ops | :yellow_circle: :six: | | | | |
| cl_pocl_svm_rect | :yellow_circle: :two: | | | | |
| cl_pocl_command_buffer_svm | :yellow_circle: :two: | | | | |
| cl_pocl_command_buffer_host_buffer | :yellow_circle: :two: | | | | |
| cl_khr_command_buffer | :yellow_circle: :two: | :yellow_circle: :two: | | | |
| cl_khr_command_buffer_multi_device | :yellow_circle: :two: | | | | | |
| cl_khr_command_buffer_mutable_dispatch | :yellow_circle: :two: | | | | |
| cl_khr_subgroups | :yellow_circle: :two: | :yellow_circle: :one: | :yellow_circle: :two: | | |
| cl_intel_spirv_subgroups | :red_circle:  | :yellow_circle: :one: | | | |
| cl_khr_subgroup_ballot | :yellow_circle: :two: | :yellow_circle: :two: | | | |
| cl_khr_subgroup_shuffle | :yellow_circle: :two: | :yellow_circle: :two: | | | |
| cl_khr_subgroup_shuffle_relative | :red_circle: | :yellow_circle: :two: | | | |
| cl_khr_subgroup_extended_types | :red_circle: | :yellow_circle: :two: | | | |
| cl_khr_subgroup_non_uniform_arithmetic | :red_circle: | :yellow_circle: :two: | | | |
| cl_khr_subgroup_non_uniform_vote | :red_circle: | :yellow_circle: :two: | | | |
| cl_khr_subgroup_clustered_reduce | :red_circle: | :yellow_circle: :two: | | | |
| cl_intel_subgroups | :yellow_circle: :two: | :yellow_circle: :two: | | | |
| cl_intel_subgroups_short | :yellow_circle: :two: | :yellow_circle: :two: | | | |
| cl_intel_subgroups_char | :yellow_circle: :two: | :yellow_circle: :two: | | | |
| cl_intel_subgroups_long | :red_circle: | :yellow_circle: :two: | | | |
| cl_intel_subgroup_local_block_io | :red_circle: | :yellow_circle: :two: | | | |
| cl_intel_required_subgroup_size | :yellow_circle: :two: | | | | |
| cl_exp_tensor | :yellow_circle: :two: | :yellow_circle: :one: | | | |
| cl_exp_defined_builtin_kernels | :yellow_circle: :two: | :yellow_circle: :one: | | | |
| cl_khr_il_program | :yellow_circle: :three: | :yellow_circle: :three: | :yellow_circle: :three: | | |
| cl_khr_spirv_queries | :yellow_circle: :three: | :yellow_circle: :three: | :yellow_circle: :three: | | |
| cl_khr_spirv_no_integer_wrap_decoration | :green_circle: | :green_circle: | | | |
| cl_khr_spirv_linkonce_odr | :red_circle: | :yellow_circle: :one: | | | |
| cl_khr_fp16 | :yellow_circle: :four: | :yellow_circle: :two: :one: | :yellow_circle: :seven: | | |
| cl_khr_fp64 | :yellow_circle: :five: | :yellow_circle: :two: :one: | :green_circle: | | |
| cl_intel_split_work_group_barrier | :red_circle: | :yellow_circle: :two: | | | |
| cl_nv_device_attribute_query |:red_circle: | :red_circle: | :green_circle: | | |

### OpenCL C features

Some of these have prequisites (e.g. for __opencl_c_ext_fp64_local_atomic_add
requires cl_khr_fp64 & cl_ext_float_atomics), these must be additionally
supported by the device.

| Features  | CPU device | Level Zero | CUDA | OpenASIP | Remote |
|:----------|:----------:|:----------:|:----:|:--------:|:------:|
| __opencl_c_images | :green_circle: | :yellow_circle: :one: | | | |
| __opencl_c_3d_image_writes | :green_circle: | :yellow_circle: :one: | | | |
| __opencl_c_atomic_order_acq_rel | :green_circle: | :yellow_circle: :one: | :green_circle: | | |
| __opencl_c_atomic_order_seq_cst | :green_circle: | :yellow_circle: :one: | :green_circle: | | |
| __opencl_c_atomic_scope_device | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_atomic_scope_all_devices | :green_circle: | :yellow_circle: :one: | | | |
| __opencl_c_generic_address_space | :green_circle: | :green_circle: | :green_circle:  | | |
| __opencl_c_work_group_collective_functions | :green_circle: | :green_circle:  | | | |
| __opencl_c_integer_dot_product_input_4x8bit |  :green_circle: | :yellow_circle: :two: :one: | | | |
| __opencl_c_integer_dot_product_input_4x8bit_packed | :green_circle: | :yellow_circle: :two: :one: | | | |
| __opencl_c_subgroups | :yellow_circle: :two: | :yellow_circle: :two: :one: | :yellow_circle: :two: | | |
| __opencl_c_read_write_images | :yellow_circle: :two: | :yellow_circle: :one: | | | |
| __opencl_c_program_scope_global_variables | :yellow_circle: :two: | :yellow_circle: :two: | :green_circle:  | | |
| __opencl_c_ext_fp32_global_atomic_add | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp32_local_atomic_add | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp32_global_atomic_min_max | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp32_local_atomic_min_max | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp64_global_atomic_add | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp64_local_atomic_add | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp64_global_atomic_min_max | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_ext_fp64_local_atomic_min_max | :green_circle: | :yellow_circle: :one: | :green_circle:  | | |
| __opencl_c_work_group_collective_functions | :red_circle: | :green_circle: | | | |


### Notes

1. these extensions are depend on Hardware and Runtime (LevelZero, CUDA) support;
   if both are available, the features are enabled by default
2. these extensions are only enabled when ENABLE_CONFORMANCE=OFF,
   because they're incomplete or fail some corner-cases or similar
3. these extensions are supported when PoCL is compiled with SPIR-V support
4. `cl_khr_fp16` is supported on CPU if all of these are met:
   * both the host & Clang compilers support the required type (_Float16)
     and can emulate / execute operations on the type
   * Note: GCC only supports _Float16 since version 12
   * LLVM >= 19, ENABLE_CONFORMANCE=OFF, Linux, CpuArch != i386
5. `cl_khr_fp64` support is enabled on CPU, unless explicitly disabled
6. `cl_khr_extended_bit_ops` only supported with LLVM 20+
7. `cl_khr_fp16` is supported on CUDA devices with Compute Capability >= 6.0


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

Note that this installs an ICD loader from KhronosGroup and the builtin
OpenCL implementation will be invisible when your application is linked
to this loader. To make both pocl and the builtin OpenCL implementaiton
visible, do

    mamba install pocl ocl_icd_wrapper_apple

## License

PoCL is distributed under the terms of the MIT license. Contributions are expected
to be made with the same terms.
