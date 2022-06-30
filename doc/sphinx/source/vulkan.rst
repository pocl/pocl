Vulkan driver
=================

This driver uses libvulkan and clspv to run OpenCL code on GPU devices via Vulkan API.

NOTE: THIS DRIVER IS INCOMPLETE, without an active maintainer. Pull Requests welcomed.

Installation
-------------

Required:

 * vulkan drivers (on Ubuntu, "mesa-vulkan-drivers" for opensource vulkan drivers)
 * vulkan development files (on Ubuntu, "vulkan-headers" and "libvulkan-dev")
 * SPIR-V tools (for clspv; on Ubuntu, package "spirv-tools")

The Vulkan headers, devices and library must support at least Vulkan version 1.1;
1.0 devices may work but are untested. With 1.0 headers, pocl-vulkan won't compile.

Optional:

 * "vulkan-validationlayers-dev" for vulkan validation layers
 * "vulkan-tools" or "vulkan-utils" package for vulkaninfo

Note that the Vulkan device MUST support the following extensions (clspv requirements):

* VK_KHR_variable_pointers
* VK_KHR_storage_buffer_storage_class
* VK_KHR_shader_non_semantic_info

Optional extensions:
 * VK_EXT_external_memory_host for CL_MEM_USE_HOST_PTR to be useful
 * VK_KHR_16bit_storage, VK_KHR_8bit_storage, VK_KHR_shader_float16_int8
   to be able to use 8 bit and 16 bit integers

Easiest to check is with vulkaninfo utility, they must be listed in 'Device Extensions' section.

To build the full pocl-vulkan, first you must build the clspv compiler::

    git clone https://github.com/google/clspv.git
    cd clspv
    python utils/fetch_sources.py
    mkdir build ; cd build
    cmake /path/to/clspv -DCLSPV_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -jX
    make install

... this will take some time and space, because it compiles its own checkout of LLVM.

After the build, copy "clspv" and "clspv-reflection" binaries to some place CLSPV_BIN_DIR

Then build the vulkan driver::

    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_VULKAN=1 -DCLSPV_DIR=${CLSPV_BIN_DIR} <path-to-pocl-source-dir>

You may set VULKAN_SDK env variable before running cmake, then it will look for libvulkan in VULKAN_SDK/lib directory.

After build, libpocl can be tested with (run in the build directory)::

     OCL_ICD_VENDORS=$PWD/ocl-vendors/pocl-tests.icd POCL_BUILDING=1 POCL_DEVICES=vulkan ./examples/example1/example1

Adding `POCL_VULKAN_VALIDATE=1 POCL_DEBUG=vulkan` into the environment enables the use of validation layers,
this will make output from PoCL much more verbose.

It is possible to build & use pocl-vulkan without clspv, but this limits the usability of the driver to clCreateProgramWithBinaries() with poclbinaries.

What works
------------

 * both integrated and discrete GPUs are supported
 * buffer (cl_mem) kernel arguments
 * POD (plain old data) kernel arguments (int32 and float32; other int/float types
   are enabled only if indicated by device features; structs with these types)
 * local memory, both as static (in-kernel) and as kernel argument
 * constant memory, both at module-scope and as kernel argument
 * most 1.2 API calls
 * CL_MEM_USE_HOST_PTR with clCreateBuffer(), if the device
   supports VK_EXT_external_memory_host
 * global offsets to clEnqueueNDRangeKernel

Doesnt work / missing
-----------------------

 * image / sampler support
 * clLinkProgram & clCompileProgram
 * clCreateBuffer(): CL_MEM_USE_HOST_PTR on dGPUs doesn't work
 * clCreateBuffer(): the CL_MEM_ALLOC_HOST_PTR flag is ignored

Unfinished / non-optimal
-------------------------

 * missing sub-allocator for small allocations
 * statically sized structs that create certain limits
 * descriptor set should be cached (setup once per kernel, then just update)
 * command buffers should be cached
 * kernel library - check what clspv is missing
 * push constants for POD arguments instead of POD UBO
 * stop using deprecated clspv-reflection, instead extract the
   kernel metadata from the SPIR-V file itself


Known Bugs
-----------

Validation layers can print this message:

"After specialization was applied, VkShaderModule 0xXY0000XY[] does not contain valid spirv for stage VK_SHADER_STAGE_COMPUTE_BIT. The Vulkan spec states: module must be a valid VkShaderModule handle (https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VUID-VkPipelineShaderStageCreateInfo-module-parameter)"

This is (AFAIK) caused by Clspv reflection metadata present in SPIR-V, and is harmless.

The pocl vulkan driver will wait indefinitely for a kernel to finish. However GPU drivers have their own "freeze detection" timeouts and could kill the kernel sooner. This would result in PoCL aborting with error -4 (device lost).

Clspv can compile a lot of code, but is still unfinished and has bugs, so pocl-vulkan may fail to compile OpenCL code.

Testing
---------

The tests that should work with Vulkan driver can be run with tools/scripts/run_vulkan_tests.

This driver was tested with these devices:

* Intel HD 530 integrated GPU
* AMD Radeon Vega 8 iGPU
* Nvidia Quadro P600 discrete GPU
* Raspberry Pi 4 + Ubuntu 22.04
