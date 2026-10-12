/* Tests for SVM.

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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>

#include "pocl_opencl.h"

#define N_ELEMENTS 16

static char GetAddrSourceCode[] = R"raw(

  __kernel void get_addr (__global int *svm_buffer,
                          __global ulong* addr) {
    for (int i = 0; i < N_ELEMENTS; ++i) {
      svm_buffer[i] += 1;
    }
    *addr = (ulong)svm_buffer;
  }
)raw";

#define STRINGIFY(X, Y) X #Y
#define SET_N_ELEMENTS(NUM) STRINGIFY("-DN_ELEMENTS=", NUM)

int TestCGSVM(cl::Platform Platform) {

  bool AllOK = true;

  try {
    cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)Platform(), 0};
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
      std::cout << "No devices with SVM coarse grain buffer capabilities found."
                << std::endl;
      return 77;
    }

    // Basics: Create a bunch of random-sized allocations and ensure their address
    // ranges do not overlap.
    constexpr size_t NumAllocs = 1000;
    constexpr size_t MaxSize = 1024*1024;

    std::mt19937 Gen(1234);
    std::uniform_int_distribution<> Distrib(1, MaxSize);

    std::map<char*, size_t> Allocs;
    for (size_t i = 0; i < NumAllocs; ++i) {
      size_t AllocSize = Distrib(Gen);

      char *Buf = (char*)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                                      AllocSize, 0);

      // If we exhaust the SVM space space, it's fine.
      // Freeing the allocations should make the remainder of the test
      // work still, unless there's a mem leak in the implementation.
      if (Buf == nullptr)
        break;

      // Check for overlap.
      for (auto& m : Allocs) {
        if (m.first <= Buf && m.first + m.second > Buf) {
          std::cerr << "An SVM allocation at " << std::hex << (size_t)Buf
                    << " with size " << std::dec << AllocSize
                    << " overlaps with a previous one at " << std::hex
                    << (size_t)m.first << " with size " << m.second << std::endl;
          return false;
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

    Program.build(SuitableDevices, SET_N_ELEMENTS(N_ELEMENTS));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    constexpr size_t BufSize = N_ELEMENTS * sizeof(int);
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

    int HostBuf[] = {0, 1, 2, 3};
    // Initialize the first inputs via an SVM memcpy command.

    // Without the destination being host-mapped...
    Queue.enqueueMemcpySVM(&CGSVMBuf[0], &HostBuf[0], CL_TRUE, 2 * sizeof(int));

    Queue.enqueueMapSVM(CGSVMBuf, CL_TRUE, CL_MAP_READ, 2 * sizeof(int));
    for (int i = 0; i < 2; ++i) {
      if (CGSVMBuf[i] != i) {
        AllOK = false;
        std::cerr << "CGSVMBuf[" << i << "] " << std::hex << &CGSVMBuf[i]
                  << " expected to be " << i << " but got " << (int)CGSVMBuf[i]
                  << std::endl;
      }
      if (HostBuf[i] != i) {
        AllOK = false;
        std::cerr << "HostBuf[" << i << "] expected to be " << i << " but got "
                  << (int)HostBuf[i] << std::endl;
      }
    }

    // ...and while it has been host-mapped.
    Queue.enqueueMemcpySVM(&CGSVMBuf[2], &HostBuf[2], CL_TRUE, 2 * sizeof(int));
    Queue.enqueueUnmapSVM(CGSVMBuf);

    // remap with Write flag
    Queue.enqueueMapSVM(CGSVMBuf, CL_TRUE, CL_MAP_WRITE, BufSize);
    // Write the rest of the inputs directly.
    for (int i = 4; i < N_ELEMENTS; ++i) {
      CGSVMBuf[i] = i;
    }
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

    // Verify some of the data with SVMMemcpy
    Queue.enqueueMemcpySVM(&HostBuf[0], &CGSVMBuf[0], CL_TRUE, 4 * sizeof(int));
    std::cerr << std::dec;
    for (int i = 0; i < 4; ++i) {
      if (HostBuf[i] != i + 1) {
        AllOK = false;
        std::cerr << "Wrong data in the memcopied buf at " << i << " expected "
                  << i + 1 << " got " << HostBuf[i] << std::endl;
      }
    }

    // Verify all data with SVMMap
    Queue.enqueueMapSVM(CGSVMBuf, true, CL_MAP_READ, BufSize);
    for (int i = 0; i < N_ELEMENTS; ++i) {
      if (CGSVMBuf[i] != i + 1) {
        AllOK = false;
        std::cerr << "CGSVMBuf[" << i << "] expected to be " << i + 1
                  << " but got " << (int)CGSVMBuf[i] << std::endl;
      }
    }
    Queue.enqueueUnmapSVM(CGSVMBuf);
    Queue.finish();

    clSVMFree(Context.get(), CGSVMBuf);

  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  if (AllOK) {
    printf("PASSED\n");
    return EXIT_SUCCESS;
  } else
    return EXIT_FAILURE;
}

int TestMultiDevice_CGSVM(cl::Platform Platform) {

  bool AllOK = true;

  try {
    cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)Platform(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        SuitableDevices.push_back(Dev);
      } else {
        std::cout << "Device '" << Dev.getInfo<CL_DEVICE_NAME>()
                  << "' doesn't support CG SVM." << std::endl;
        continue;
      }
    }

    if (SuitableDevices.size() < 2) {
      std::cout << "At least 2 devices with SVM coarse grain buffer "
                   "capabilities needed."
                << std::endl;
      return 77;
    }

    // Test that an allocation has the same virtual address in each device and
    // the host.
    int *CGSVMBuf = (int *)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE,
                                        sizeof(cl_int) * N_ELEMENTS, 0);
    if (CGSVMBuf == nullptr) {
      std::cerr << "CG SVM allocation returned a nullptr." << std::endl;
      return EXIT_FAILURE;
    }

    cl::Program::Sources Sources({GetAddrSourceCode});
    cl::Program Program(Context, Sources);

    Program.build(SuitableDevices, SET_N_ELEMENTS(N_ELEMENTS));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    std::map<cl::Device *, cl::CommandQueue *> Queues;
    for (cl::Device &Device : SuitableDevices) {
      cl::CommandQueue *Q = new cl::CommandQueue(Context, Device, 0);
      Queues[&Device] = Q;
    }

    // Check that the SVM addresses of the SVM buf are the same in
    // the device side.
    for (cl::Device &Device : SuitableDevices) {
      cl::CommandQueue &Queue = *Queues[&Device];

      cl_ulong AddrFromKernel = 1;

      cl::Buffer AddrCLBuffer =
          cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr);

      ::clSetKernelArgSVMPointer(GetAddrKernel.get(), 0, CGSVMBuf);
      // Now the runtime should automatically migrate the buffer to device 1
      // input -> device 2 input etc. due to the SVM argument reference.

      GetAddrKernel.setArg(1, AddrCLBuffer);
      Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                                 cl::NullRange);
      Queue.enqueueReadBuffer(AddrCLBuffer,
                              CL_TRUE, // block
                              0, sizeof(cl_ulong), (void *)&AddrFromKernel);
      Queue.finish();
      if (CGSVMBuf != (void *)AddrFromKernel) {
        std::cerr << "SVM buffer's device address on kernel and host "
                     "sides do not match. Host sees "
                  << std::hex << CGSVMBuf << " while a device sees "
                  << AddrFromKernel << std::endl;
        AllOK = false;
      }
    }

    // Check the buffer migration to and between devices with
    // an event synch.
    // The runtime should automatically migrate the buffer to device 1 input ->
    // device 2 input etc. due to the SVM argument reference.
    cl::Event PrevKernelExecutionEvent;

    cl::CommandQueue &Q = *(*Queues.begin()).second;

    // Initialize the SVM buf at the host side.
    Q.enqueueMapSVM(CGSVMBuf, true, CL_MAP_WRITE, N_ELEMENTS * sizeof(cl_int));
    for (int i = 0; i < N_ELEMENTS; ++i) {
      CGSVMBuf[i] = i + 1;
    }
    Q.enqueueUnmapSVM(CGSVMBuf);

    for (cl::Device &Device : SuitableDevices) {
      cl::CommandQueue &Queue = *Queues[&Device];

      cl_ulong AddrFromKernel = 1;

      cl::Buffer AddrCLBuffer =
          cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr);

      ::clSetKernelArgSVMPointer(GetAddrKernel.get(), 0, CGSVMBuf);

      GetAddrKernel.setArg(1, AddrCLBuffer);

      std::vector<cl::Event> WaitEvents;
      if (&Device != &*(SuitableDevices.begin()))
        WaitEvents.push_back(PrevKernelExecutionEvent);

      Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                                 cl::NullRange, &WaitEvents,
                                 &PrevKernelExecutionEvent);
      Queue.enqueueReadBuffer(AddrCLBuffer,
                              CL_TRUE, // block
                              0, sizeof(cl_ulong), (void *)&AddrFromKernel);
    }

    cl::CommandQueue &LastQ = *Queues[&SuitableDevices.back()];
    LastQ.finish();

    // After all devices have finished execution, check the kernel check the
    // result.
    LastQ.enqueueMapSVM(CGSVMBuf, true, CL_MAP_READ,
                        N_ELEMENTS * sizeof(cl_int));
    for (int i = 0; i < N_ELEMENTS; ++i) {
      int Expected = i + 1 + Devices.size();
      if (CGSVMBuf[i] != Expected) {
        std::cerr << "SVM buffer element " << i
                  << " was not incremented by all devices "
                     "as expected: got "
                  << CGSVMBuf[i] << " expected " << Expected << std::endl;
        AllOK = false;
      }
    }
    LastQ.enqueueUnmapSVM(CGSVMBuf);
    LastQ.finish();

    for (cl::Device &Device : SuitableDevices) {
      delete Queues[&Device];
    }
    // TODO: run with multiple devices in the PoCL suite

  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << " (" << err.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }

  if (AllOK) {
    printf("PASSED\n");
    return EXIT_SUCCESS;
  } else
    return EXIT_FAILURE;
}

static char SimpleKernelSourceCode[] = R"raw(

  __kernel void simple_kernel(__global int *Out,
                              __global int *In) {
    *Out = *In;
  }
)raw";

// OpenCL version of simple_kernel.hip in the chipStar samples.
int TestSimpleKernel_CGSVM(cl::Platform Platform) {
  bool AllOK = true;

  try {
    cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)Platform(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        SuitableDevices.push_back(Dev);
        break;
      } else {
        std::cout << "Device '" << Dev.getInfo<CL_DEVICE_NAME>()
                  << "' doesn't support CG SVM." << std::endl;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with SVM coarse grain buffer capabilities found."
                << std::endl;
      return 77;
    }

    cl::CommandQueue Queue(Context, SuitableDevices[0], 0);

    cl::Program::Sources Sources({SimpleKernelSourceCode});
    cl::Program Program(Context, Sources);

    Program.build(SuitableDevices);

    cl::Kernel SimpleKernel(Program, "simple_kernel");

    int InH = 123, OutH = 0, *InD, *OutD;
    OutD =
        (int *)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE, sizeof(int), 0);
    InD = (int *)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE, sizeof(int), 0);

    if (OutD == nullptr || InD == nullptr) {
      std::cerr << "Unable to allocate SVM buffers.\n";
      return EXIT_FAILURE;
    }

    CHECK_CL_ERROR(::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, InD, &InH,
                                        sizeof(int), 0, nullptr, nullptr));

    ::clSetKernelArgSVMPointer(SimpleKernel.get(), 0, OutD);
    ::clSetKernelArgSVMPointer(SimpleKernel.get(), 1, InD);

    Queue.enqueueNDRangeKernel(SimpleKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);

    CHECK_CL_ERROR(::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &OutH, OutD,
                                        sizeof(int), 0, nullptr, nullptr));

    clSVMFree(Context.get(), OutD);
    clSVMFree(Context.get(), InD);

    if (OutH == 123) {
      printf("PASSED\n");
    } else {
      AllOK = false;
      printf("OutH=%d\n", OutH);
    }
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  if (AllOK)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

// Test for cl_mem-wrapped SVM pointers.
int TestCLMem_SVM(cl::Platform Platform) {
  cl_int Err = 0;
  bool AllOK = true;

  try {
    cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)Platform(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        SuitableDevices.push_back(Dev);
        break;
      } else {
        std::cout << "Device '" << Dev.getInfo<CL_DEVICE_NAME>()
                  << "' doesn't support CG SVM." << std::endl;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with SVM coarse grain buffer capabilities found."
                << std::endl;
      return 77;
    }

    cl::CommandQueue Q(Context, SuitableDevices[0], 0);

    cl::Program::Sources Sources({SimpleKernelSourceCode});
    cl::Program Program(Context, Sources);

    Program.build(SuitableDevices);

    cl::Kernel SimpleKernel(Program, "simple_kernel");

    int InH = 123, OutH = 0, *InD, *OutD;
    InD = (int *)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE, sizeof(int), 0);
    OutD =
        (int *)::clSVMAlloc(Context.get(), CL_MEM_READ_WRITE, sizeof(int), 0);
    if (OutD == nullptr || InD == nullptr) {
      std::cerr << "Unable to allocate SVM buffers.\n";
      return EXIT_FAILURE;
    }

    cl::Buffer clBufInD(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                        sizeof(int), InD, &Err);

    CHECK_CL_ERROR(Err);

    cl::Buffer clBufOutD(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         sizeof(int), OutD, &Err);
    CHECK_CL_ERROR(Err);

    if (!clBufInD.getInfo<CL_MEM_USES_SVM_POINTER>() ||
        !clBufOutD.getInfo<CL_MEM_USES_SVM_POINTER>()) {
      std::cerr << "cl_mem wrappers for the SVM pointers do not have "
                   "CL_MEM_USES_SVM_POINTER set.\n";
      return EXIT_FAILURE;
    }

    // Now clWriteBuffer() should allow us to update the SVM
    // region as an alternative to clEnqueueSVMMemcpy().
    CHECK_CL_ERROR(
        Q.enqueueWriteBuffer(clBufInD, CL_FALSE, 0, sizeof(int), &InH));

    // We still should be able to use clSetKernelArgSVMPointer
    // ::clSetKernelArgSVMPointer(SimpleKernel.get(), 0, OutD);
    // ...or clSetKernelArg()
    CHECK_CL_ERROR(SimpleKernel.setArg(0, clBufOutD));
    CHECK_CL_ERROR(SimpleKernel.setArg(1, clBufInD));

    CHECK_CL_ERROR(Q.enqueueNDRangeKernel(SimpleKernel, cl::NullRange,
                                          cl::NDRange(1), cl::NullRange));

    CHECK_CL_ERROR(
        Q.enqueueReadBuffer(clBufOutD, CL_TRUE, 0, sizeof(int), &OutH));
    clSVMFree(Context.get(), OutD);
    clSVMFree(Context.get(), InD);

    if (OutH == 123) {
      printf("PASSED\n");
    } else {
      AllOK = false;
      printf("FAILED OutH=%d\n", OutH);
    }
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  if (AllOK)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

int TestFGSVM(cl::Platform Platform) {

  bool AllOK = true;

  try {
    cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)Platform(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
        SuitableDevices.push_back(Dev);
        break;
      } else {
        std::cout << "Device '" << Dev.getInfo<CL_DEVICE_NAME>()
                  << "' doesn't support FG SVM." << std::endl;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with SVM fine grain buffer capabilities found."
                << std::endl;
      return 77;
    }

    // Basics: Create a bunch of random-sized allocations and ensure their
    // address ranges do not overlap.
    constexpr size_t NumAllocs = 1000;
    constexpr size_t MaxSize = 1024 * 1024;

    std::mt19937 Gen(1234);
    std::uniform_int_distribution<> Distrib(1, MaxSize);

    std::map<char *, size_t> Allocs;
    for (size_t i = 0; i < NumAllocs; ++i) {
      size_t AllocSize = Distrib(Gen);

      char *Buf = (char *)::clSVMAlloc(
          Context.get(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
          AllocSize, 0);

      // If we exhaust the SVM space space, it's fine.
      // Freeing the allocations should make the remainder of the test
      // work still, unless there's a mem leak in the implementation
      // side.
      if (Buf == nullptr)
        break;

      // Check for overlap.
      for (auto &m : Allocs) {
        if (m.first <= Buf && m.first + m.second > Buf) {
          std::cerr << "An SVM allocation at " << std::hex << (size_t)Buf
                    << " with size " << std::dec << AllocSize
                    << " overlaps with a previous one at " << std::hex
                    << (size_t)m.first << " with size " << m.second
                    << std::endl;
          return false;
        }
      }
      Allocs[Buf] = AllocSize;
    }

    if (Allocs.size() == 0) {
      std::cerr << "Unable to allocate any SVM chunks." << std::endl;
      return EXIT_FAILURE;
    }
    for (auto &m : Allocs) {
      // std::cout << "Freeing " << std::hex << (size_t)m.first << std::endl;
      clSVMFree(Context.get(), m.first);
    }

    cl::CommandQueue Queue(Context, SuitableDevices[0], 0);

    cl::Program::Sources Sources({GetAddrSourceCode});
    cl::Program Program(Context, Sources);

    Program.build(SuitableDevices, SET_N_ELEMENTS(N_ELEMENTS));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    constexpr size_t BufSize = N_ELEMENTS * sizeof(int);
    int *FGSVMBuf = (int *)::clSVMAlloc(
        Context.get(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        BufSize, 0);

    if (FGSVMBuf == nullptr) {
      std::cerr << "FG SVM allocation returned a nullptr." << std::endl;
      return false;
    }

    cl_ulong AddrFromKernel = 1;

    cl::Buffer AddrCLBuffer =
        cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr);

    ::clSetKernelArgSVMPointer(GetAddrKernel.get(), 0, FGSVMBuf);
    GetAddrKernel.setArg(1, AddrCLBuffer);

    int HostBuf[] = {0, 1, 2, 3};
    // Initialize the first inputs via an SVM memcpy command.

    // Without the destination being host-mapped...
    CHECK_CL_ERROR(::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &FGSVMBuf[0],
                                        &HostBuf[0], 2 * sizeof(int), 0,
                                        nullptr, nullptr));

    for (int i = 0; i < 2; ++i) {
      if (FGSVMBuf[i] != i) {
        AllOK = false;
        std::cerr << "FGSVMBuf[" << i << "] " << std::hex << &FGSVMBuf[i]
                  << " expected to be " << i << " but got " << (int)FGSVMBuf[i]
                  << std::endl;
      }
      if (HostBuf[i] != i) {
        AllOK = false;
        std::cerr << "HostBuf[" << i << "] expected to be " << i << " but got "
                  << (int)HostBuf[i] << std::endl;
      }
    }

    ::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &FGSVMBuf[2], &HostBuf[2],
                         2 * sizeof(int), 0, nullptr, nullptr);

    // Write the rest of the inputs directly.
    for (int i = 4; i < N_ELEMENTS; ++i) {
      FGSVMBuf[i] = i;
    }

    Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);
    Queue.enqueueReadBuffer(AddrCLBuffer,
                            CL_TRUE, // block
                            0, sizeof(cl_ulong), (void *)&AddrFromKernel);

    if (FGSVMBuf != (void *)AddrFromKernel) {
      std::cerr << "FG buffer's device address on kernel side and host "
                   "side do not match. Host sees "
                << std::hex << FGSVMBuf
                << " while "
                   "the device sees "
                << AddrFromKernel << std::endl;
      AllOK = false;
    }

    // Read some of the data with SVMMemcpy().
    ::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &HostBuf[0], &FGSVMBuf[0],
                         4 * sizeof(int), 0, nullptr, nullptr);

    std::cerr << std::dec;
    for (int i = 0; i < N_ELEMENTS; ++i) {
      if (FGSVMBuf[i] != i + 1) {
        AllOK = false;
        std::cerr << "FGSVMBuf[" << i << "] expected to be " << i + 1
                  << " but got " << (int)FGSVMBuf[i] << std::endl;
      }
      if (i < 4 && i + 1 != HostBuf[i]) {
        AllOK = false;
        std::cerr << "Wrong data in the memcopied buf at " << i << " expected "
                  << i + 1 << " got " << HostBuf[i] << std::endl;
      }
    }
    clSVMFree(Context.get(), FGSVMBuf);
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  if (!AllOK)
    return EXIT_FAILURE;
  else
    return EXIT_SUCCESS;
}

int TestSSVM(cl::Platform Platform) {

  bool AllOK = true;

  try {
    cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)Platform(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<cl::Device> SuitableDevices;

    for (cl::Device &Dev : Devices) {
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
        SuitableDevices.push_back(Dev);
        break;
      } else {
        std::cout << "Device '" << Dev.getInfo<CL_DEVICE_NAME>()
                  << "' doesn't support FG System SVM." << std::endl;
      }
    }

    if (SuitableDevices.empty()) {
      std::cout << "No devices with fine grain system SVM capabilities found."
                << std::endl;
      return 77;
    }

    cl::CommandQueue Queue(Context, SuitableDevices[0], 0);

    cl::Program::Sources Sources({GetAddrSourceCode});
    cl::Program Program(Context, Sources);

    Program.build(SuitableDevices, SET_N_ELEMENTS(N_ELEMENTS));

    cl::Kernel GetAddrKernel(Program, "get_addr");

    constexpr size_t BufSize = N_ELEMENTS * sizeof(int);
    int *SSVMBuf = (int *)::malloc(BufSize);

    if (SSVMBuf == nullptr) {
      std::cerr << "malloc() returned a nullptr." << std::endl;
      return false;
    }

    cl_ulong AddrFromKernel = 1;

    cl::Buffer AddrCLBuffer =
        cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr);

    ::clSetKernelArgSVMPointer(GetAddrKernel.get(), 0, SSVMBuf);
    GetAddrKernel.setArg(1, AddrCLBuffer);

    int HostBuf[] = {0, 1, 2, 3};
    // Initialize the first inputs via an SVM memcpy command.

    // Without the destination being host-mapped...
    CHECK_CL_ERROR(::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &SSVMBuf[0],
                                        &HostBuf[0], 2 * sizeof(int), 0,
                                        nullptr, nullptr));

    for (int i = 0; i < 2; ++i) {
      if (SSVMBuf[i] != i) {
        AllOK = false;
        std::cerr << "SSVMBuf[" << i << "] " << std::hex << &SSVMBuf[i]
                  << " expected to be " << i << " but got " << (int)SSVMBuf[i]
                  << std::endl;
      }
      if (HostBuf[i] != i) {
        AllOK = false;
        std::cerr << "HostBuf[" << i << "] expected to be " << i << " but got "
                  << (int)HostBuf[i] << std::endl;
      }
    }

    ::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &SSVMBuf[2], &HostBuf[2],
                         2 * sizeof(int), 0, nullptr, nullptr);

    // Write the rest of the inputs directly.
    for (int i = 4; i < N_ELEMENTS; ++i) {
      SSVMBuf[i] = i;
    }

    Queue.enqueueNDRangeKernel(GetAddrKernel, cl::NullRange, cl::NDRange(1),
                               cl::NullRange);
    Queue.enqueueReadBuffer(AddrCLBuffer,
                            CL_TRUE, // block
                            0, sizeof(cl_ulong), (void *)&AddrFromKernel);

    if (SSVMBuf != (void *)AddrFromKernel) {
      std::cerr << "FG system buffer's device address on kernel side and host "
                   "side do not match. Host sees "
                << std::hex << SSVMBuf
                << " while "
                   "the device sees "
                << AddrFromKernel << std::endl;
      AllOK = false;
    }

    // Read some of the data with SVMMemcpy().
    ::clEnqueueSVMMemcpy(Queue.get(), CL_TRUE, &HostBuf[0], &SSVMBuf[0],
                         4 * sizeof(int), 0, nullptr, nullptr);

    std::cerr << std::dec;
    for (int i = 0; i < N_ELEMENTS; ++i) {
      if (SSVMBuf[i] != i + 1) {
        AllOK = false;
        std::cerr << "SSVMBuf[" << i << "] expected to be " << i + 1
                  << " but got " << (int)SSVMBuf[i] << std::endl;
      }
      if (i < 4 && i + 1 != HostBuf[i]) {
        AllOK = false;
        std::cerr << "Wrong data in the memcopied buf at " << i << " expected "
                  << i + 1 << " got " << HostBuf[i] << std::endl;
      }
    }

    free(SSVMBuf);
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  if (!AllOK)
    return EXIT_FAILURE;
  else
    return EXIT_SUCCESS;
}

int TestSSVMFill(cl::Platform Platform) {

  bool AllOK = true;

  try {
    std::vector<cl::Device> Devices;
    Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);

    // A context of only this device: in a mixed context, SVM pointers are
    // validated against the context's SVM allocation device, which might
    // not support fine-grained system SVM.
    cl::Device Device;
    for (cl::Device &Dev : Devices) {
      if (Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
          CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
        Device = Dev;
        break;
      }
    }

    if (Device() == nullptr) {
      std::cout << "No devices with fine grain system SVM capabilities found."
                << std::endl;
      return 77;
    }

    cl::Context Context(Device);
    cl::CommandQueue Queue(Context, Device, 0);

    std::vector<int> SSVMBuf(N_ELEMENTS, -1);

    // Fill all but the first and last elements. The fill waits on a user
    // event and the pattern is overwritten before it runs, so it must be
    // copied at enqueue time.
    int Pattern = 42;
    cl::UserEvent Gate(Context);
    cl_event GateEvent = Gate.get();
    cl_event FillEvent = nullptr;
    CHECK_CL_ERROR(::clEnqueueSVMMemFill(
        Queue.get(), &SSVMBuf[1], &Pattern, sizeof(int),
        (N_ELEMENTS - 2) * sizeof(int), 1, &GateEvent, &FillEvent));
    Pattern = 0;
    Gate.setStatus(CL_COMPLETE);
    CHECK_CL_ERROR(::clWaitForEvents(1, &FillEvent));

    cl_command_type FillType = 0;
    CHECK_CL_ERROR(::clGetEventInfo(FillEvent, CL_EVENT_COMMAND_TYPE,
                                    sizeof(FillType), &FillType, nullptr));
    ::clReleaseEvent(FillEvent);
    if (FillType != CL_COMMAND_SVM_MEMFILL) {
      AllOK = false;
      std::cerr << "SVM fill event has command type " << std::hex << FillType
                << std::dec << std::endl;
    }

    for (int i = 0; i < N_ELEMENTS; ++i) {
      int Expected = (i == 0 || i == N_ELEMENTS - 1) ? -1 : 42;
      if (SSVMBuf[i] != Expected) {
        AllOK = false;
        std::cerr << "SSVMBuf[" << i << "] expected to be " << Expected
                  << " but got " << SSVMBuf[i] << std::endl;
      }
    }

    if (::clEnqueueSVMMemFill(Queue.get(), SSVMBuf.data(), nullptr, sizeof(int),
                              sizeof(int), 0, nullptr,
                              nullptr) != CL_INVALID_VALUE) {
      AllOK = false;
      std::cerr << "SVM fill with a NULL pattern was not rejected" << std::endl;
    }
  } catch (cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
    AllOK = false;
  }

  if (!AllOK)
    return EXIT_FAILURE;
  else
    return EXIT_SUCCESS;
}

int main() {

  std::vector<cl::Platform> PlatformList;

  cl::Platform::get(&PlatformList);

  std::cout << "TestSimpleKernel_CGSVM: ";
  if (TestSimpleKernel_CGSVM(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "TestCLMem_SVM: ";
  if (TestCLMem_SVM(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "TestCGSVM: ";
  if (TestCGSVM(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "TestFGSVM: ";
  if (TestFGSVM(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "TestSSVM: ";
  if (TestSSVM(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "TestSSVMFill: ";
  if (TestSSVMFill(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  std::cout << "TestMultiDevice_CGSVM: ";
  if (TestMultiDevice_CGSVM(PlatformList[0]) == EXIT_FAILURE)
    return EXIT_FAILURE;

  PlatformList[0].unloadCompiler();

  std::cout << "OK" << std::endl;

  return EXIT_SUCCESS;
}
