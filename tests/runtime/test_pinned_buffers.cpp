/* Test the pinned buffers extension.

   Copyright (c) 2023 Pekka Jääskeläinen / Intel Finland Oy

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

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS

#include "../../include/CL/cl_ext_pocl.h"
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include "pocl_opencl.h"

#define BUF_SIZE 16

static char GetAddrSourceCode[] = R"raw(

  __kernel void get_addr (__global int *pinned_buffer,
                          __global ulong* addr) {
    for (int i = 0; i < BUF_SIZE; ++i)
      pinned_buffer[i] += 1;
    *addr = (ulong)pinned_buffer;
  }
)raw";

void *getDeviceAddressFromHost(cl::Buffer &Buf) {
  cl_mem_pinning Pinning;

  Buf.getInfo(CL_MEM_DEVICE_PTRS, &Pinning);
  return Pinning.address;
}

int main(void) {

  unsigned Errors = 0;
  bool AllOK = true;

  try {
    std::vector<cl::Platform> PlatformList;

    cl::Platform::get(&PlatformList);

    cl_context_properties cprops[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(PlatformList[0])(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      std::string Exts = Dev.getInfo<CL_DEVICE_EXTENSIONS>();
      if (Exts.find("cl_pocl_pinned_buffers") != std::string::npos) {
        SuitableDevices.push_back(Dev);
        break;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with cl_pocl_pinned_buffers found.";
      return 77;
    }
    int PinnedBufferHost[BUF_SIZE];
    int PinnedBufferHost2[BUF_SIZE];

    for (int i = 0; i < BUF_SIZE; ++i) {
      PinnedBufferHost[i] = i;
      PinnedBufferHost2[i] = i + 1;
    }

    cl_ulong DeviceAddrFromKernel = 1;

    cl::CommandQueue Queue(Context, SuitableDevices[0], 0);

    cl::Program::Sources Sources({GetAddrSourceCode});
    cl::Program Program(Context, Sources);

#define STRINGIFY(X, Y) X #Y
#define SET_BUF_SIZE(NUM) STRINGIFY("-DBUF_SIZE=", NUM)

    Program.build(SuitableDevices, SET_BUF_SIZE(BUF_SIZE));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    cl::Buffer PinnedCLBuffer = cl::Buffer(
        Context, CL_MEM_READ_WRITE | CL_MEM_PINNED | CL_MEM_COPY_HOST_PTR,
        BUF_SIZE * sizeof(cl_int), (void *)&PinnedBufferHost[0]);

    if (getDeviceAddressFromHost(PinnedCLBuffer) == nullptr) {
      std::cerr << "Pinned buffers should get allocated immediately to get the "
                   "address assigned."
                << std::endl;
      return EXIT_FAILURE;
    }

    cl::Buffer AddrCLBuffer =
        cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr);

    GetAddrKernel.setArg(0, PinnedCLBuffer);
    GetAddrKernel.setArg(1, AddrCLBuffer);

    Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);

    Queue.enqueueReadBuffer(PinnedCLBuffer,
                            CL_TRUE, // block
                            0, BUF_SIZE * sizeof(cl_int),
                            (void *)&PinnedBufferHost[0]);

    Queue.enqueueReadBuffer(AddrCLBuffer,
                            CL_TRUE, // block
                            0, sizeof(cl_ulong), (void *)&DeviceAddrFromKernel);

    AllOK = true;
    for (int i = 0; i < BUF_SIZE; ++i) {
      if (PinnedBufferHost[i] != i + 1) {
        AllOK = false;
        std::cerr << "PinnedBufferHost[" << i << "] expected to be " << i + 1
                  << " but got " << PinnedBufferHost[i] << std::endl;
      }
    }

    if (getDeviceAddressFromHost(PinnedCLBuffer) !=
        (void *)DeviceAddrFromKernel) {
      std::cerr << "Pinned buffer's device address on kernel side and host "
                   "side do not match"
                << std::endl;
      return EXIT_FAILURE;
    }

    // Test a buffer which doesn't have any hostptr associated with it.
    cl::Buffer PinnedCLBufferNoHostCopy =
        cl::Buffer(Context, CL_MEM_PINNED, BUF_SIZE * sizeof(cl_int), nullptr);

    GetAddrKernel.setArg(0, PinnedCLBufferNoHostCopy);

    Queue.enqueueWriteBuffer(PinnedCLBufferNoHostCopy,
                             CL_TRUE, // block
                             0, BUF_SIZE * sizeof(cl_int),
                             (void *)&PinnedBufferHost[0]);

    Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);

    Queue.enqueueReadBuffer(PinnedCLBufferNoHostCopy,
                            CL_TRUE, // block
                            0, BUF_SIZE * sizeof(cl_int),
                            (void *)&PinnedBufferHost2[0]);

    Queue.enqueueReadBuffer(AddrCLBuffer,
                            CL_TRUE, // block
                            0, sizeof(cl_ulong), (void *)&DeviceAddrFromKernel);

    for (int i = 0; i < BUF_SIZE; ++i) {
      if (PinnedBufferHost2[i] != i + 2) {
        AllOK = false;
        std::cerr << "PinnedBufferHost2[" << i << "] expected to be " << i + 2
                  << " but got " << PinnedBufferHost2[i] << std::endl;
      }
    }

    if (getDeviceAddressFromHost(PinnedCLBufferNoHostCopy) !=
        (void *)DeviceAddrFromKernel) {
      std::cerr << "Pinned buffer's device address on kernel side and host "
                   "side do not match"
                << std::endl;
      return EXIT_FAILURE;
    }

  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }

  CHECK_CL_ERROR (clUnloadCompiler ());

  if (AllOK) {
    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
  } else
    return EXIT_FAILURE;
}
