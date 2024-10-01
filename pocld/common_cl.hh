/* common_cl.hh -- helper types that wrap opencl.hpp handles and metadata

   Copyright (c) 2018 Michal Babej / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef POCL_REMOTE_COMMON_CL_HH
#define POCL_REMOTE_COMMON_CL_HH

#include "CL/cl.h"
#include "CL/opencl.hpp"

#include "messages.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

typedef std::unique_ptr<cl::Buffer> clBufferPtr;
typedef std::unique_ptr<cl::Sampler> clSamplerPtr;
typedef std::unique_ptr<cl::Image> clImagePtr;

typedef std::unique_ptr<cl::Kernel> clKernelPtr;
typedef std::vector<PoclRemoteArgType> clKernelArgTypeVector;
typedef std::unique_ptr<clKernelArgTypeVector> clKernelArgTypeVectorPtr;

typedef struct clKernelMetadata {
  KernelMetaInfo_t meta;
  std::vector<ArgumentInfo_t> arg_meta;
} clKernelMetadata;

typedef std::vector<void *> clFakeKernelPtrArgs;
typedef std::vector<std::vector<char>> clFakeKernelArgsPOD;

typedef int (*fakeBuiltinKernelCallback)(void *SharedCLContext, void *side_thr,
                                         cl::Context *ctx, cl::CommandQueue *cq,
                                         cl::Kernel *ker,
                                         clFakeKernelPtrArgs &ptr_args,
                                         clFakeKernelArgsPOD &pod_args,
                                         std::vector<cl::Event> *dependencies,
                                         cl::Event *event);

typedef std::vector<PoclRemoteArgType> clKernelArgTypeVector;
typedef struct clKernelStruct {
  std::vector<cl::Kernel> perDeviceKernels;
  clFakeKernelPtrArgs fakeKernelPtrArgs;
  clFakeKernelArgsPOD fakeKernelPODArgs;
  clKernelMetadata *metaData;
  fakeBuiltinKernelCallback callback;
  unsigned numArgs = 0;
  bool isFakeBuiltin = false;
  std::mutex Lock;
} clKernelStruct;
typedef std::unique_ptr<clKernelStruct> clKernelStructPtr;

typedef std::unique_ptr<cl::Program> clProgramPtr;
typedef struct clProgramStruct {
  clProgramPtr uptr;
  std::vector<cl::Device> devices;
  std::vector<cl::Kernel> prebuilt_kernels;
  std::vector<clKernelMetadata> kernel_meta;
  unsigned numKernels = 0;
  bool isFakeBuiltin = false;
} clProgramStruct;

typedef std::unique_ptr<clProgramStruct> clProgramStructPtr;

typedef std::shared_ptr<cl::CommandQueue> clCommandQueuePtr;
typedef std::unique_ptr<cl::CommandQueue> clCommandQueueUPtr;

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
