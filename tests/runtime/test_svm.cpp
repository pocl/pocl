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
#include <map>
#include <random>

#include "pocl_opencl.h"

#define BUF_SIZE 16

static char GetAddrSourceCode[] = R"raw(

  __kernel void get_addr (__global int *svm_buffer,
                          __global ulong* addr) {
    for (int i = 0; i < BUF_SIZE; ++i) {
      svm_buffer[i] += 1;
      printf("kern-side svm_buffer[%d] == %d\n", i, svm_buffer[i]);
    }
    *addr = (ulong)svm_buffer;
    printf("kern-side addr %p\n", svm_buffer);
  }
)raw";

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
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        SuitableDevices.push_back(Dev);
        break;
      } else {
        std::cout << "Device '" << Dev.getInfo<CL_DEVICE_NAME>() << "' doesn't support CG SVM."
                  << std::endl;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with SVM coarse grain buffer capabilities found.";
      return 77;
    }

    // Basics: Create a bunch of random-sized allocations and ensure their address
    // ranges do not overlap.
    constexpr size_t NumAllocs = 1000;
    constexpr size_t MaxSize = 1024*1024;

    std::mt19937 Gen(1234);
    std::uniform_int_distribution<> Distrib(1, MaxSize);

    std::map<char*, size_t> Allocs;
    for (int i = 0; i < NumAllocs; ++i) {
      size_t AllocSize = Distrib(Gen);

      char *Buf = (char*)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                                      AllocSize, 0);

      // If we exhaust the SVM space space, it's fine.
      // Freeing the allocations should make the remainder of the test
      // work still, unless there's a mem leak in the implementation
      // side.
      if (Buf == nullptr)
        break;

      // Check for overlap.
      for (auto& m : Allocs) {
        if (m.first <= Buf && m.first + m.second > Buf) {
          std::cerr << "An SVM allocation at " << std::hex << (size_t)Buf
                    << " with size " << std::dec << AllocSize
                    << " overlaps with a previous one at " << std::hex
                    << (size_t)m.first << " with size " << m.second << std::endl;
          return EXIT_FAILURE;
        }
      }
      Allocs[Buf] = AllocSize;
    }

    if (Allocs.size() == 0) {
      std::cerr << "Unable to allocate any SVM chunks." << std::endl;
      return EXIT_FAILURE;
    }
    for (auto& m : Allocs) {
      // std::cout << "Freeing " << std::hex << (size_t)m.first << std::endl;
      clSVMFree(Context.get(), m.first);
    }

    cl::CommandQueue Queue(Context, SuitableDevices[0], 0);

    cl::Program::Sources Sources({GetAddrSourceCode});
    cl::Program Program(Context, Sources);

#define STRINGIFY(X, Y) X #Y
#define SET_BUF_SIZE(NUM) STRINGIFY("-DBUF_SIZE=", NUM)

    Program.build(SuitableDevices, SET_BUF_SIZE(BUF_SIZE));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    constexpr size_t BufSize = BUF_SIZE * sizeof(int);
    int *CGSVMBuf = (int*)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                                       BufSize, 0);

    if (CGSVMBuf == nullptr) {
      std::cerr << "CG SVM allocation returned a nullptr." << std::endl;
      return EXIT_FAILURE;
    }

    cl_ulong AddrFromKernel = 1;

    cl::Buffer AddrCLBuffer =
        cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr);

    ::clSetKernelArgSVMPointer(GetAddrKernel.get(), 0, CGSVMBuf);
    GetAddrKernel.setArg(1, AddrCLBuffer);

    Queue.enqueueMapSVM(CGSVMBuf, true, CL_MAP_WRITE, BufSize);

    for (int i = 0; i < BUF_SIZE; ++i) {
      CGSVMBuf[i] = i;
    }

    // Is this actually unneccessary because the kernel execution and finishing are
    // syncpoints?
    Queue.enqueueUnmapSVM(CGSVMBuf);
    Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);

    Queue.enqueueReadBuffer(AddrCLBuffer,
                            CL_TRUE, // block
                            0, sizeof(cl_ulong), (void *)&AddrFromKernel);

    if (CGSVMBuf != (void *)AddrFromKernel) {
      std::cerr << "CG buffer's device address on kernel side and host "
        "side do not match. Host sees " << std::hex << CGSVMBuf << " while "
        "the device sees " << AddrFromKernel << std::endl;
      AllOK = false;
    }

    // Is not needed because the kernel execution and finishing are
    // syncpoints.
    Queue.enqueueMapSVM(CGSVMBuf, CL_TRUE, CL_MAP_READ, BufSize);

    for (int i = 0; i < BUF_SIZE; ++i) {
      if (CGSVMBuf[i] != i + 1) {
        AllOK = false;
        std::cerr << "CGSVMBuf[" << i << "] expected to be " << i + 1
                  << " but got " << (int)CGSVMBuf[i] << std::endl;
      }
    }
    clSVMFree(Context.get(), CGSVMBuf);

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
