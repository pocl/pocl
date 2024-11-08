/* Tests for subbuffers, especially their legal concurrent access patterns.

   Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>

static char VecAddSrc[] = R"raw(
  __kernel void vecadd (const __global int *A, const __global int *B,
                        __global int *C) {
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
  }
)raw";

#if 0
    printf("gid %u A %d B %d C %d\n",
            get_global_id (0), A[get_global_id(0)], B[get_global_id(0)],
            C[get_global_id(0)]);
#endif

// Split a buffer to N subbuffers which are concurrently processed
// by multiple kernel commands in different command queues.
// Ideally the sub-buffers are processed in parallel since there are
// no dependencies communicated by the client code to the runtime.
int TestOutputDataDecomposition() {

  unsigned Errors = 0;
  bool AllOK = true;

  try {
    std::vector<cl::Platform> PlatformList;

    cl::Platform::get(&PlatformList);

    cl_context_properties cprops[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(PlatformList[0])(), 0};
    cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, cprops);

    std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();

    if (Devices.empty()) {
      std::cout << "No devices found." << std::endl;
      return EXIT_FAILURE;
    }

    const size_t NumParallelQueues = std::max((size_t)2, Devices.size());
    const size_t WorkShare =
      (Devices[0].getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / sizeof(cl_int)) *
      4;
    // Leave uncovered space at the end of subbuffer to make the end address
    // unaligned to regression test the corner case of user subbuffer uncovered
    // parts ending before an unaligned address.
    const size_t Uncovered = 13;
    // Leave one chunk of the data untouched so we can check that
    // the migrations are done at subbuffer level.
    const size_t NumData = (NumParallelQueues + 1) * WorkShare;

    std::cerr << "Number of devices: " << Devices.size() << std::endl;
    std::cerr << "Total data (bytes) == " << NumData * sizeof(cl_int)
              << std::endl;
    std::cerr << "WorkShare (bytes) == " << WorkShare * sizeof(cl_int)
              << std::endl;
    std::cerr << "Processing data before "
              << NumParallelQueues * WorkShare * sizeof(cl_int) << " bytes "
              << std::endl;
    std::cerr << "Last sub-buffer starts at byte offset "
              << (NumParallelQueues - 1) * WorkShare * sizeof(cl_int)
              << std::endl;

    std::vector<int> HostBufA, HostBufB, HostBufC;
    for (size_t i = 0; i < NumData; ++i) {
      HostBufA.push_back(i);
      HostBufB.push_back(0);
      HostBufC.push_back(3);
    }

    cl::Buffer ABuffer =
        cl::Buffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(cl_int) * NumData, HostBufA.data());

    cl::Buffer BBuffer =
        cl::Buffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(cl_int) * NumData, HostBufB.data());

    cl::Buffer CBuffer =
        cl::Buffer(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                   sizeof(cl_int) * NumData, HostBufC.data());

    cl::Program::Sources Sources({VecAddSrc});
    cl::Program Program(Context, Sources);
    Program.build(Devices);
    cl::Kernel VecAddKernel(Program, "vecadd");

    std::vector<cl::Buffer> SubBuffers;
    std::vector<cl::CommandQueue> Queues;
    std::vector<cl::Event> KernelEvents;
    // Spawn a bunch of kernel commands in their independent command queues
    // (which could target different devices) to process their piece of the
    // data.
    for (size_t i = 0; i < NumParallelQueues; ++i) {

      cl::CommandQueue Queue(Context, Devices[i % Devices.size()], 0);
      Queues.push_back(Queue);

      cl_buffer_region Region;
      Region.origin = i * WorkShare * sizeof(cl_int);
      Region.size = (WorkShare - Uncovered) * sizeof(cl_int);

      std::cout << "Q" << i << " sub-buffer origin element " << i * WorkShare
                << " last element "
                << i * WorkShare + (WorkShare - Uncovered - 1) << std::endl;
      cl::Buffer ASubBuffer =
          ABuffer.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region);
      cl::Buffer BSubBuffer =
          BBuffer.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region);

      VecAddKernel.setArg(0, ASubBuffer);
      VecAddKernel.setArg(1, BSubBuffer);

      cl::Buffer CSubBuffer =
          CBuffer.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region);
      VecAddKernel.setArg(2, CSubBuffer);

      SubBuffers.push_back(ASubBuffer);
      SubBuffers.push_back(BSubBuffer);
      SubBuffers.push_back(CSubBuffer);

      // Initialize the sub-buffers using fillbuffer, just to test it out.
      cl::Event FEv1, FEv2;

      Queue.enqueueFillBuffer(BSubBuffer, (cl_int)2, 0,
                              (WorkShare - Uncovered) * sizeof(cl_int), nullptr,
                              &FEv1);

      Queue.enqueueFillBuffer(CSubBuffer, (cl_int)1, 0,
                              (WorkShare - Uncovered) * sizeof(cl_int), nullptr,
                              &FEv2);

      cl::vector<cl::Event> Deps;
      Deps.push_back(FEv1);
      Deps.push_back(FEv2);

      cl::Event Ev;
      Queue.enqueueNDRangeKernel(VecAddKernel, cl::NullRange,
                                 cl::NDRange(WorkShare - Uncovered),
                                 cl::NullRange, &Deps, &Ev);
      KernelEvents.push_back(Ev);
    }

    std::vector<int> AfterSubBufCContents(NumData);

    Queues[0].enqueueReadBuffer(CBuffer, CL_FALSE, 0, sizeof(cl_int) * NumData,
                                AfterSubBufCContents.data(), &KernelEvents);

    // Push a kernel that reads and writes the whole buffer.
    VecAddKernel.setArg(0, CBuffer);
    // Should add 2 to all elements again, in place.
    VecAddKernel.setArg(1, BBuffer);
    VecAddKernel.setArg(2, CBuffer);

    // The event dep on the previous kernel commands should ensure the data is
    // implicitly migrated to the parent buffer.
    Queues[0].enqueueNDRangeKernel(VecAddKernel, cl::NullRange,
                                   cl::NDRange(WorkShare * NumParallelQueues),
                                   cl::NullRange, &KernelEvents, nullptr);

    std::vector<int> NewBufCContents(NumData);

    Queues[0].enqueueReadBuffer(CBuffer, CL_FALSE, 0, sizeof(cl_int) * NumData,
                                NewBufCContents.data());

    // Push a kernel that inputs the last CSubbuffer, that should get updated
    // with the changes done by the previous command.
    VecAddKernel.setArg(0, SubBuffers.back());
    VecAddKernel.setArg(1, BBuffer); // Should add 2 to all elements, in place.
    VecAddKernel.setArg(2, SubBuffers.back());

    Queues[0].enqueueNDRangeKernel(VecAddKernel, cl::NullRange,
                                   cl::NDRange(WorkShare - Uncovered),
                                   cl::NullRange, &KernelEvents, nullptr);

    std::vector<int> FinalBufCContents(NumData);

    Queues[0].enqueueReadBuffer(CBuffer, CL_FALSE, 0, sizeof(cl_int) * NumData,
                                FinalBufCContents.data());

    Queues[0].finish();

    // Check the data after the parallel sub-buffer launches.
    for (size_t i = 0; i < NumData; ++i) {
      int Share = i / WorkShare;
      if (i >= WorkShare * Share &&
          i < WorkShare * Share + WorkShare - Uncovered &&
          i < WorkShare * NumParallelQueues) {
        if (AfterSubBufCContents[i] != i + 2) {
          std::cerr << "ERROR: after sub-bufs " << i << " was "
                    << AfterSubBufCContents[i] << " expected " << i + 2
                    << std::endl;
          AllOK = false;
          break;
        }
      } else {
        // The last part and the unaligned gaps after the sub-buffers should
        // remain untouched.
        if (AfterSubBufCContents[i] != 3) {
          std::cerr << "ERROR: after sub-bufs the last part " << i << " was "
                    << AfterSubBufCContents[i] << " expected 3\n";
          AllOK = false;
          break;
        }
      }
    }

    // Check the data before the last kernel launch.
    for (size_t i = 0; i < NumData; ++i) {
      int Share = i / WorkShare;
      if (i >= WorkShare * Share &&
          i < WorkShare * Share + WorkShare - Uncovered &&
          i < WorkShare * NumParallelQueues) {
        if (NewBufCContents[i] != i + 2 + 2) {
          std::cerr << "ERROR: " << i << " was " << NewBufCContents[i]
                    << " expected " << i + 2 + 2 << std::endl;
          AllOK = false;
          break;
        }
      } else {
        // The last part should remain untouched.
        if (NewBufCContents[i] != 3) {
          std::cerr << "ERROR: " << i << " was " << NewBufCContents[i]
                    << " expected 3\n";
          AllOK = false;
          break;
        }
      }
    }

    // In the final state there should be one additional 2 addition in the
    // last manipulated part of the array.
    for (size_t i = 0; i < NumData; ++i) {
      int Share = i / WorkShare;
      if (i >= WorkShare * Share &&
          i < WorkShare * Share + WorkShare - Uncovered &&
          i < WorkShare * NumParallelQueues) {
        if (i < (WorkShare * (NumParallelQueues - 1))) {
          if (FinalBufCContents[i] != i + 2 + 2) {
            std::cerr << "ERROR: final " << i << " was " << FinalBufCContents[i]
                      << " expected " << i + 2 + 2 << std::endl;
            AllOK = false;
            break;
          }
        } else if (i < (WorkShare * NumParallelQueues)) {
          // The part which was not touched by sub-buffers.
          if (FinalBufCContents[i] != i + 2 + 2 + 2) {
            std::cerr << "ERROR: final " << i << " was " << FinalBufCContents[i]
                      << " expected " << i + 2 + 2 << std::endl;
            AllOK = false;
            break;
          }
        }
      } else {
        // The very last part should still remain untouched.
        if (FinalBufCContents[i] != 3) {
          std::cerr << "ERROR: final last part " << i << " was "
                    << FinalBufCContents[i] << " expected 3\n";
          AllOK = false;
          break;
        }
      }
    }

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

int main() {

  std::cout << "TestOutputDataDecomposition: ";
  if (TestOutputDataDecomposition() == EXIT_FAILURE)
    return EXIT_FAILURE;

  CHECK_CL_ERROR(clUnloadCompiler());

  std::cout << "OK" << std::endl;

  return EXIT_SUCCESS;
}
