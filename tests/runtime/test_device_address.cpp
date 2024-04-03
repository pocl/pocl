/* Test the cl_ext_buffer_device_address extension.

   Copyright (c) 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_opencl.h"

#include "../../include/CL/cl_ext_pocl.h"
#include <CL/opencl.hpp>

#include <cassert>
#include <cstdlib>
#include <iostream>

#define BUF_SIZE 16

// A kernel that gets the device-seen address of the buffer.
static char GetAddrSourceCode[] = R"raw(

  __kernel void get_addr (__global int *buffer,
                          __global ulong* addr) {
    for (int i = 0; i < BUF_SIZE; ++i)
      buffer[i] += 1;
    *addr = (ulong)buffer;
  }
)raw";

// A kernel that accesses another buffer indirectly.
static char IndirectAccess[] = R"raw(

  __kernel void indirect_access (__global long* in_addr,
                                 __global int* out) {
    *out = **(int __global* __global*)in_addr;
  }
)raw";

// A kernel that gets passed a pointer to a middle of a buffer,
// with the data _before_ the passed pointer. Tests the property
// of sub-buffers to synchronize the whole parent buffer when
// using the CL_MEM_BUFFER_DEVICE_ADDRESS flag.
static char PtrArith[] = R"raw(

  __kernel void ptr_arith (__global int* in_addr,
                           __global int* out) {
    *out = *(in_addr - 1);
  }
)raw";

void *getDeviceAddressFromHost(cl::Buffer &Buf) {
  void *Addr;
  cl_int Err = Buf.getInfo(CL_MEM_DEVICE_PTR_EXT, &Addr);

  if (Err != CL_SUCCESS) {
    std::cerr << "Got error " << Err
              << " when asking for CL_MEM_DEVICE_PTR_EXT\n";
    return nullptr;
  }

  return Addr;
}

int main(void) {

  unsigned Errors = 0;
  bool AllOK = true;

  try {
    std::vector<cl::Platform> PlatformList;

    cl::Platform::get(&PlatformList);

    cl::Platform SelectedPlatform;
    bool PlatformFound = false;
    for (cl::Platform &Platform : PlatformList) {
      if (Platform.getInfo<CL_PLATFORM_EXTENSIONS>().find(
              "cl_ext_buffer_device_address") == std::string::npos)
        continue;
      SelectedPlatform = Platform;
      PlatformFound = true;
      break;
    }

    if (!PlatformFound) {
      std::cerr << "No platforms with cl_ext_buffer_device_address found. Not "
                   "testing PoCL?\n";
      return EXIT_FAILURE;
    }

    cl_context_properties cprops[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(SelectedPlatform)(), 0};

    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      std::string Exts = Dev.getInfo<CL_DEVICE_EXTENSIONS>();
      std::cout << Dev.getInfo<CL_DEVICE_NAME>() << " "
                << Dev.getInfo<CL_DEVICE_VERSION>() << ": ";
      if (Exts.find("cl_ext_buffer_device_address") != std::string::npos) {
        std::cout << "suitable" << std::endl;
        SuitableDevices.push_back(Dev);
        break;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with cl_ext_buffer_device_address found.\n";
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

    cl::Program::Sources Sources({GetAddrSourceCode, IndirectAccess, PtrArith});
    cl::Program Program(Context, Sources);

#define STRINGIFY(X, Y) X #Y
#define SET_BUF_SIZE(NUM) STRINGIFY("-DBUF_SIZE=", NUM)

    Program.build(SuitableDevices, SET_BUF_SIZE(BUF_SIZE));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    cl::Buffer PinnedCLBuffer = cl::Buffer(
        Context,
        (cl_mem_flags)(CL_MEM_READ_WRITE | CL_MEM_DEVICE_ADDRESS_EXT |
                       CL_MEM_COPY_HOST_PTR),
        (size_t)BUF_SIZE * sizeof(cl_int), (void *)&PinnedBufferHost[0]);

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
      std::cerr << "Pinned buffer's device address on the kernel side and "
                << "the host side do not match" << std::endl;
      return EXIT_FAILURE;
    }

    // Test a buffer which doesn't have any hostptr associated with it.
    cl::Buffer PinnedCLBufferNoHostCopy = cl::Buffer(
        Context, CL_MEM_DEVICE_ADDRESS_EXT, BUF_SIZE * sizeof(cl_int), nullptr);

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

    // Test a buffer which is passed to the kernel indirectly.
    cl::Kernel IndirectAccessKernel(Program, "indirect_access");

    int DataIn = 1234;
    // A devaddr buffer with the payload data.
    cl::Buffer DevAddrCLBuffer = cl::Buffer(
        Context,
        (cl_mem_flags)(CL_MEM_READ_WRITE | CL_MEM_DEVICE_ADDRESS_EXT |
                       CL_MEM_COPY_HOST_PTR),
        sizeof(int), (void *)&DataIn);

    void *DevAddr = getDeviceAddressFromHost(DevAddrCLBuffer);

    // A basic buffer used to pass the other buffer's address.
    cl::Buffer NormalCLBufferIn = cl::Buffer(
        Context, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
        sizeof(cl_long), (void *)&DevAddr);

    cl::Buffer NormalCLBufferOut = cl::Buffer(
        Context, (cl_mem_flags)(CL_MEM_WRITE_ONLY), sizeof(cl_int), nullptr);

    IndirectAccessKernel.setArg(0, NormalCLBufferIn);
    IndirectAccessKernel.setArg(1, NormalCLBufferOut);
    if (::clSetKernelExecInfo(IndirectAccessKernel.get(),
                              CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT,
                              sizeof(void *), &DevAddr) != CL_SUCCESS) {
      std::cerr << "Setting indirect access for device ptrs failed!\n";
      return EXIT_FAILURE;
    }

    /// The Level 0 doesn't get the buffer initialized with
    /// CL_MEM_COPY_HOST_PTR. This is a workaround until that is fixed.
    Queue.enqueueWriteBuffer(DevAddrCLBuffer,
                             CL_TRUE, // block
                             0, sizeof(cl_int), (void *)&DataIn);

    Queue.enqueueNDRangeKernel(IndirectAccessKernel, cl::NullRange,
                               cl::NDRange(1), cl::NullRange);

    int DataOut = -1;
    Queue.enqueueReadBuffer(NormalCLBufferOut,
                            CL_TRUE, // block
                            0, sizeof(cl_int), (void *)&DataOut);

    if (DataIn != DataOut) {
      AllOK = false;
      std::cerr << "Passing data via indirect buffers failed. Got: " << DataOut
                << " expected: " << DataIn << "\n";
      return EXIT_FAILURE;
    }

    // Test using clSetKernelArgDevicePointerEXT to pass pointers to
    // inside a buffer.
    cl::Kernel PtrArithKernel(Program, "ptr_arith");

    clSetKernelArgDevicePointerEXT_fn clSetKernelArgDevicePointer =
        (clSetKernelArgDevicePointerEXT_fn)
            clGetExtensionFunctionAddressForPlatform(
                SelectedPlatform(), "clSetKernelArgDevicePointerEXT");

    assert(clSetKernelArgDevicePointer != nullptr);

    clSetKernelArgDevicePointer(
        PtrArithKernel.get(), 0,
        (cl_uint *)getDeviceAddressFromHost(PinnedCLBuffer) + 2);
    PtrArithKernel.setArg(1, NormalCLBufferOut);

    DataOut = -1;

    Queue.enqueueNDRangeKernel(PtrArithKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);

    Queue.enqueueReadBuffer(NormalCLBufferOut,
                            CL_TRUE, // block
                            0, sizeof(cl_int), (void *)&DataOut);

    if (DataOut != PinnedBufferHost[1]) {
      AllOK = false;
      std::cerr << "Negative offsetting from passed in pointer failed: "
                << "Expected: " << PinnedBufferHost[1] << " got: " << DataOut
                << "\n";
      return EXIT_FAILURE;
    }

  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  CHECK_CL_ERROR (clUnloadCompiler ());

  if (AllOK) {
    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
  } else
    return EXIT_FAILURE;
}
