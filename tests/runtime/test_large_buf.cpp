/* Tests a case where buffers are >32bit large (mainly intended for Level Zero)

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_opencl.h"

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>

#define INPUT_BUFFER_SIZE (5UL << 30)
#define OUTPUT_BUFFER_SIZE 128

#define WORKITEMS 1048576

// (INPUT_BUFFER_SIZE / WORKITEMS)
#define LOOPSIZE 5120
#define BUILD_OPTS "-DLOOPSIZE=5120"

const char *SOURCE = R"RAW(
__kernel void test_kernel2 (global const uchar    *in,
                            global       ulong   *out)
{
  size_t gid = get_global_id(0) * LOOPSIZE;
  ulong sum = 0;
  for (unsigned i = 0; i < LOOPSIZE; ++i) {
    sum += in[gid + i];
  }
  atomic_fetch_add((volatile global atomic_ulong *)out, (ulong)sum);
};
)RAW";

int main(void) {
  std::vector<cl::Platform> PlatformList;
  bool Ok = false;
  try {

    // Pick platform
    cl::Platform::get(&PlatformList);
    cl::Platform UsedPlatform = PlatformList[0];

    // Pick first platform
    cl_context_properties CtxProps[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(UsedPlatform()), 0};
    cl::Context Context(CL_DEVICE_TYPE_ALL, CtxProps);

    // Query the set of devices attched to the context
    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device UsedDevice;

    for (auto &Dev : Devices) {
      cl_ulong MaxAlloc = Dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
      if (MaxAlloc > INPUT_BUFFER_SIZE) {
        UsedDevice = Dev;
        break;
      }
    }

    if (UsedDevice() == nullptr) {
      std::cout << "No device with sufficiently large MaxMemAlloc found.\n"
                << std::endl;
      return 77;
    } else {
      std::cout << "Using device " << UsedDevice.getInfo<CL_DEVICE_NAME>()
                << " with MaxMemAllocSize: "
                << UsedDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << "\n";
    }

    unsigned char *Inputs = new unsigned char[INPUT_BUFFER_SIZE];
    cl_ulong Sum = 0;

    std::cout << "Generating random input...\n";
    for (cl_ulong i = 0; i < (INPUT_BUFFER_SIZE / 4); i++) {
      unsigned r = rand();
      cl_ulong offset = i * 4;
      memcpy(Inputs + offset, &r, sizeof(r));
      Sum += Inputs[offset];
      Sum += Inputs[offset + 1];
      Sum += Inputs[offset + 2];
      Sum += Inputs[offset + 3];
    }
    std::cout << "... done\n";

    // Create and program from source
    cl::Program::Sources Sources({SOURCE});
    cl::Program Program(Context, Sources);

    // Build program
    Program.build(UsedDevice, BUILD_OPTS);

    cl::Buffer InBuffer = cl::Buffer(Context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR |
                                         CL_MEM_ALLOC_HOST_PTR,
                                     INPUT_BUFFER_SIZE, (void *)&Inputs[0]);

    cl::Buffer OutBuffer =
        cl::Buffer(Context, CL_MEM_WRITE_ONLY, OUTPUT_BUFFER_SIZE);

    cl::Kernel Kernel(Program, "test_kernel2");

    Kernel.setArg(0, InBuffer);
    Kernel.setArg(1, OutBuffer);

    cl::CommandQueue Queue(Context, Devices[0], 0);
    // clear output buffer
    cl_ulong Zeroes = 0;
    Queue.enqueueFillBuffer(OutBuffer, Zeroes, 0, OUTPUT_BUFFER_SIZE);

    Queue.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange(WORKITEMS),
                               cl::NullRange);

    cl_ulong *Output =
        (cl_ulong *)Queue.enqueueMapBuffer(OutBuffer,
                                           CL_TRUE, // block
                                           CL_MAP_READ, 0, OUTPUT_BUFFER_SIZE);
    // verify
    Ok = (Output[0] == Sum);

    Queue.enqueueUnmapMemObject(OutBuffer, (void *)Output);

    Queue.finish();
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }

  PlatformList[0].unloadCompiler();

  if (Ok) {
    std::cout << "Compare OK" << std::endl;
    return EXIT_SUCCESS;
  } else {
    std::cout << "FAIL: result mismatch" << std::endl;
    return EXIT_FAILURE;
  }
}
