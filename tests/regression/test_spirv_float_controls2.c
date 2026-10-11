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
  A kernel using SPV_KHR_float_controls2 to mark a multiply and an add as
  contractable (FPFastMathMode AllowContract), which base SPIR-V cannot
  express. Producers such as LLVM's SPIR-V back-end emit this for IR with the
  `contract` fast-math flag (e.g. Julia's `muladd`); without the extension,
  they can only drop the flag, and the multiply and add are never fused.

  The SPIR-V reader rejects modules using extensions that are not enabled
  through --spirv-ext, so before the extension was enabled, building the
  program failed. Contraction is allowed but not required, so either the fused
  or the unfused result is correct. Test source
  (test_spirv_float_controls2.spv):

               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability FloatControls2
               OpExtension "SPV_KHR_float_controls2"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %kernel "muladd" %gid
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %gid Constant
               OpDecorate %mul FPFastMathMode AllowContract
               OpDecorate %add FPFastMathMode AllowContract
      %ulong = OpTypeInt 64 0
    %v3ulong = OpTypeVector %ulong 3
  %ptr_input = OpTypePointer Input %v3ulong
       %void = OpTypeVoid
      %float = OpTypeFloat 32
 %ptr_global = OpTypePointer CrossWorkgroup %float
         %fn = OpTypeFunction %void %ptr_global %ptr_global %ptr_global
        %gid = OpVariable %ptr_input Input
     %kernel = OpFunction %void None %fn
          %c = OpFunctionParameter %ptr_global
          %a = OpFunctionParameter %ptr_global
          %b = OpFunctionParameter %ptr_global
      %entry = OpLabel
       %gid3 = OpLoad %v3ulong %gid Aligned 32
          %i = OpCompositeExtract %ulong %gid3 0
      %a_ptr = OpInBoundsPtrAccessChain %ptr_global %a %i
      %b_ptr = OpInBoundsPtrAccessChain %ptr_global %b %i
      %c_ptr = OpInBoundsPtrAccessChain %ptr_global %c %i
      %a_val = OpLoad %float %a_ptr Aligned 4
      %b_val = OpLoad %float %b_ptr Aligned 4
      %c_val = OpLoad %float %c_ptr Aligned 4
        %mul = OpFMul %float %a_val %b_val
        %add = OpFAdd %float %mul %c_val
               OpStore %c_ptr %add Aligned 4
               OpReturn
               OpFunctionEnd
*/

#include "pocl_opencl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

#define N 16

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

  size_t extensions_size;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, 0, NULL,
                              &extensions_size));
  const char **extensions = malloc(extensions_size);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR,
                              extensions_size, extensions, NULL));
  int supported = 0;
  for (size_t i = 0; i < extensions_size / sizeof(const char *); i++)
    if (strcmp(extensions[i], "SPV_KHR_float_controls2") == 0)
      supported = 1;
  free(extensions);
  if (!supported) {
    printf("SKIP: The test requires SPV_KHR_float_controls2\n");
    return 77;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  FILE *f = fopen(SRCDIR "/test_spirv_float_controls2.spv", "rb");
  if (!f) {
    printf("Failed to open test_spirv_float_controls2.spv\n");
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

  cl_kernel kernel = clCreateKernel(program, "muladd", &err);
  CHECK_ERROR(err);

  // a * b = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so the unfused result is
  // 0 while the fused one keeps the 2^-24.
  float a[N], b[N], c[N];
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = 1.0f + ldexpf(1.0f, -12);
    c[i] = -(1.0f + ldexpf(1.0f, -11));
  }
  float fused = fmaf(a[0], b[0], c[0]);
  volatile float product = a[0] * b[0];
  float unfused = product + c[0];

  cl_mem c_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(c), c, &err);
  CHECK_ERROR(err);
  cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(a), a, &err);
  CHECK_ERROR(err);
  cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(b), b, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_buf));

  size_t global_size = N, local_size = 8;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof(c), c, 0,
                                  NULL, NULL));

  int failed = 0;
  for (int i = 0; i < N; i++) {
    if (c[i] != fused && c[i] != unfused) {
      printf("c[%d] = %a, expected %a (fused) or %a (unfused)\n", i, c[i],
             fused, unfused);
      failed = 1;
    }
  }

  clReleaseMemObject(a_buf);
  clReleaseMemObject(b_buf);
  clReleaseMemObject(c_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  if (failed)
    return 1;
  printf("OK\n");
  return 0;
}
