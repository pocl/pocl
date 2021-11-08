Vulkan driver
=================

This driver uses libvulkan and clspv to run OpenCL code on GPU devices via Vulkan API.
Note that the Vulkan device MUST support the following extensions:

* VK_KHR_variable_pointers
* VK_KHR_storage_buffer_storage_class
* VK_KHR_shader_non_semantic_info

To build the full pocl-vulkan, first you must build the clspv compiler::

    git clone https://github.com/google/clspv.git
    cmake /path/to/clspv
    make -jX

... this will take some time and space (~20G), because it compiles its own checkout of LLVM.
After the build, copy "clspv" and "clspv-reflection" to some place CLSPV_BIN_DIR

Then build the vulkan driver::

    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_EXPERIMENTAL_DRIVERS=1 -DENABLE_VULKAN=1 -DCLSPV=${CLSPV_BIN_DIR}/clspv <path-to-pocl-source-dir>

You may set VULKAN_SDK env variable before running cmake, then it will look for libvulkan in VULKAN_SDK/lib directory.

It is possible to use pocl-vulkan without clspv, but this limits the usability of the driver to clCreateProgramWithBinaries()
with poclbinaries.

The full list of supported / unsupported features is listed in lib/CL/devices/vulkan/pocl-vulkan.c, at the start of file.

The tests that should work with Vulkan driver, can be run with tools/scripts/run_vulkan_tests.
