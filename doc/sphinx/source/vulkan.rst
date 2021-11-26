Vulkan driver
=================

NOTE: This driver is incomplete without an active maintainer. Pull Requests welcomed.



This driver uses libvulkan and clspv to run OpenCL code on GPU devices via Vulkan API.

Installation
-------------

Required:

 * vulkan drivers (on Ubuntu, "mesa-vulkan-drivers" for opensource vulkan drivers)
 * vulkan development files (on Ubuntu, "vulkan-headers" and "libvulkan-dev")
 * SPIR-V tools (for clspv; on Ubuntu, package "spirv-tools")

Optional:

 * "vulkan-validationlayers-dev" for vulkan validation layers
 * "vulkan-tools" or "vulkan-utils" package for vulkaninfo

Note that the Vulkan device MUST support the following extensions (clspv requirements):

* VK_KHR_variable_pointers
* VK_KHR_storage_buffer_storage_class
* VK_KHR_shader_non_semantic_info

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

    cmake -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_LLVM=0 -DENABLE_EXPERIMENTAL_DRIVERS=1 -DENABLE_VULKAN=1 -DCLSPV_DIR=${CLSPV_BIN_DIR} <path-to-pocl-source-dir>

You may set VULKAN_SDK env variable before running cmake, then it will look for libvulkan in VULKAN_SDK/lib directory.

After build, libpocl can be tested with::

     POCL_BUILDING=1 POCL_DEVICES=vulkan ./examples/example1/example1

Adding `POCL_VULKAN_VALIDATE=1 POCL_DEBUG=vulkan` into the environment enables the use of validation layers,
this will make output from PoCL much more verbose.

It is possible to build & use pocl-vulkan without clspv, but this limits the usability of the driver to clCreateProgramWithBinaries() with poclbinaries.

What works
------------

 * both integrated and discrete GPUs
 * buffer arguments (cl_mem)
 * POD (plain old data) arguments (int32 and float32; others are untested)
 * local memory, both as static (in-kernel) and as kernel argument
 * clEnqueue{Map,Unmap,Read,Write}Buffer & clEnqueueNDRangeKernel should work
 * clGetDeviceInfo should work

Doesnt work / missing
-----------------------

 * constant memory
 * module scope constants
 * image / sampler support
 * clCreateBuffer() with CL_MEM_USE_HOST_PTR is broken,
   the CL_MEM_ALLOC_HOST_PTR flag is ignored
 * in Vulkan, there is a device limit on max WG count;
   (the amount of workgroups that can be executed by a single command)
   - the driver needs to handle global size > than that
 * clEnqueue{Read,Write,Copy}BufferRect and clEnqueueFillBuffer
   APIs are not implemented
 * global offsets in clEnqueueNDRangeKernel() are ignored

Unfinished / non-optimal
-------------------------

 * missing sub-allocator for small allocations
 * statically sized structs that create certain limits
 * descriptor set should be cached (setup once per kernel, then just update)
 * command buffers should be cached
 * global offsets of kernel enqueue are ignored (should be solved by
   compiling two versions of each program, one with goffsets and one
   without, then select at runtime which to use)
 * some things that are stored per-kernel should be stored per-program,
   and v-v (e.g. compiled shader)
 * kernel library - check what clspv is missing
 * push constants for POD arguments instead of POD UBO
 * stop using deprecated clspv-reflection, instead extract the
   kernel metadata from the SPIR-V file itself


Known Bugs
-----------

Validation layers on Nvidia print this message:

"After specialization was applied, VkShaderModule 0xXY0000XY[] does not contain valid spirv for stage VK_SHADER_STAGE_COMPUTE_BIT. The Vulkan spec states: module must be a valid VkShaderModule handle (https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VUID-VkPipelineShaderStageCreateInfo-module-parameter)"

This seems to be harmless (AFAICT).


The kernel execution timeout is set to 60 seconds in PoCL, note however that GPU drivers have their own "freeze detection" timeouts and could kill the kernel sooner. This would result in PoCL aborting with error -4 (device lost).


Testing
---------

The tests that should work with Vulkan driver can be run with tools/scripts/run_vulkan_tests.

Devices where this driver was tested:

* Intel HD 530 integrated GPU
* AMD Vega 56 discrete GPU
* Nvidia Quadro P600 discrete GPU
