// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
  A kernel that branches on an undefined condition (found while reducing a
  Julia kernel with spirv-reduce, which introduced the undef;
  https://github.com/JuliaGPU/OpenCL.jl/issues/509):

      kernel void kernel() {
        if (undef)
          callee();
      }

  Branching on undef is undefined behavior, so LLVM reduces the kernel body to
  a single unreachable instruction. ConvertUnreachablesToReturns then deleted
  that entry block, as it deletes other predecessor-less blocks ending in an
  unreachable, which turned the kernel into a declaration that the work-group
  function could not inline, leaving a call to the deleted kernel behind.

  The crash happened when building the work-group function, so completing the
  build and launch is the test. Test source (test_unreachable_kernel.spv):

               OpCapability Kernel
               OpCapability Addresses
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %kernel "kernel"
       %void = OpTypeVoid
       %bool = OpTypeBool
         %fn = OpTypeFunction %void
      %undef = OpUndef %bool
     %callee = OpFunction %void None %fn
          %1 = OpLabel
               OpReturn
               OpFunctionEnd
     %kernel = OpFunction %void None %fn
      %entry = OpLabel
               OpBranchConditional %undef %then %join
       %then = OpLabel
          %2 = OpFunctionCall %void %callee
               OpBranch %join
       %join = OpLabel
               OpReturn
               OpFunctionEnd
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

int main(int argc, char **argv) {
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi(argv[1]) : 0;
  cl_int err;

#ifndef _WIN32
  // Fail quickly instead of tripping the much larger ctest timeout in case
  // a regressed build hangs.
  alarm(300);
#endif

  cl_uint num_platforms;
  CHECK_ERROR(clGetPlatformIDs(0, NULL, &num_platforms));
  if (platform_index >= num_platforms) {
    printf("Platform index %u out of range\n", platform_index);
    return 1;
  }
  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  CHECK_ERROR(clGetPlatformIDs(num_platforms, platforms, NULL));
  cl_platform_id platform = platforms[platform_index];
  free(platforms);

  cl_device_id device;
  CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  if (!poclu_device_supports_il(device, "SPIR-V_1.0")) {
    printf("SKIP: The test requires support for SPIR-V 1.0\n");
    return 77;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  FILE *f = fopen(SRCDIR "/test_unreachable_kernel.spv", "rb");
  if (!f) {
    printf("Failed to open test_unreachable_kernel.spv\n");
    return 1;
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *binary = malloc(size);
  if (fread(binary, 1, size, f) != size) {
    printf("Failed to read SPIR-V\n");
    free(binary);
    fclose(f);
    return 1;
  }
  fclose(f);

  cl_program program = clCreateProgramWithIL(context, binary, size, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clBuildProgram(program, 1, &device, NULL, NULL, NULL));

  cl_kernel kernel = clCreateKernel(program, "kernel", &err);
  CHECK_ERROR(err);

  size_t global_size = 16, local_size = 8;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clFinish(queue));

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  printf("OK\n");
  return 0;
}
