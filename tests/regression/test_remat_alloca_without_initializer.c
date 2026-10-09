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
  Reduced from a Julia (KernelAbstractions.jl) kernel with a @synchronize,
  compiled with bounds checks. A value computed on only some paths before a
  barrier (a PHI with an undef incoming value) is used after the barrier:

      if (n <= 1) return;
      if (lid > 1) return;
      m = ctx[4];
      if (n >= 1)
        v = 2*m - 1;          // v is undef on the other path
      barrier();
      if (!(v < ULONG_MAX))
        printf("ok");

  LLVM turns this into a PHI of the comparison result with an undef incoming
  value, which PHIsToAllocas turns into an alloca with a store of the value and
  a store of undef. After WorkitemLoops saved the alloca to the context array for
  one parallel region, the only store left on the alloca itself was the undef
  one, which addContextSaveRestore skips when looking for an initializer to
  rematerialize. With no initializer, it dereferenced a null InitializerStore.

  The shape is fragile: removing the unused lid-dependent instructions makes
  the problem disappear, so the reduced module is kept as-is.

  The crash happened when building the work-group function, so completing the
  build and launch is the test. Test source (test_remat_alloca_without_initializer.spv):

  ; Reduced from a KernelAbstractions.jl kernel with a @synchronize, compiled with bounds checks.
                 OpCapability Kernel
                 OpCapability Addresses
                 OpCapability Int64
                 OpCapability Int8
            %std = OpExtInstImport "OpenCL.std"
                 OpMemoryModel Physical64 OpenCL
                 OpEntryPoint Kernel %kernel "kernel" %lid_var
                 OpDecorate %fmt Constant
                 OpDecorate %lid_var BuiltIn LocalInvocationId
           %uint = OpTypeInt 32 0
          %uchar = OpTypeInt 8 0
           %void = OpTypeVoid
          %ulong = OpTypeInt 64 0
           %bool = OpTypeBool
        %v3ulong = OpTypeVector %ulong 3
      %ptr_in_v3 = OpTypePointer Input %v3ulong
        %ptr_uc  = OpTypePointer CrossWorkgroup %uchar
        %ptr_ul  = OpTypePointer CrossWorkgroup %ulong
      %kernel_fn = OpTypeFunction %void %ptr_ul
        %ulong_1 = OpConstant %ulong 1
        %ulong_2 = OpConstant %ulong 2
       %ulong_32 = OpConstant %ulong 32
      %ulong_max = OpConstant %ulong 18446744073709551615
         %uint_2 = OpConstant %uint 2
       %uint_784 = OpConstant %uint 784
         %uint_3 = OpConstant %uint 3
       %uchar_0  = OpConstantNull %uchar
      %uchar_111 = OpConstant %uchar 111
      %uchar_107 = OpConstant %uchar 107
      %str_type  = OpTypeArray %uchar %uint_3
      %ptr_str   = OpTypePointer UniformConstant %str_type
       %str_init = OpConstantComposite %str_type %uchar_111 %uchar_107 %uchar_0
            %fmt = OpVariable %ptr_str UniformConstant %str_init
        %lid_var = OpVariable %ptr_in_v3 Input
          %undef = OpUndef %ulong
         %kernel = OpFunction %void None %kernel_fn
            %ctx = OpFunctionParameter %ptr_ul
          %entry = OpLabel
         %ctx_uc = OpBitcast %ptr_uc %ctx
            %lid = OpLoad %v3ulong %lid_var Aligned 32
          %lid_x = OpCompositeExtract %ulong %lid 0
         %ctx_ul = OpBitcast %ptr_ul %ctx_uc
              %n = OpLoad %ulong %ctx_ul Aligned 8
        %ctx_off = OpInBoundsPtrAccessChain %ptr_uc %ctx_uc %ulong_32
         %small  = OpUGreaterThanEqual %bool %ulong_1 %n
                 OpBranchConditional %small %exit_a %check_lid
      %check_lid = OpLabel
         %shl    = OpShiftLeftLogical %ulong %lid_x %ulong_1
         %sub    = OpISub %ulong %ulong_1 %lid_x
        %lid_oob = OpUGreaterThan %bool %lid_x %ulong_1
                 OpBranchConditional %lid_oob %exit_b %check_n
        %check_n = OpLabel
         %m_ptr  = OpBitcast %ptr_ul %ctx_off
              %m = OpLoad %ulong %m_ptr Aligned 8
          %valid = OpSLessThanEqual %bool %ulong_1 %n
                 OpBranchConditional %valid %compute %merge
        %compute = OpLabel
             %m2 = OpIMul %ulong %m %ulong_2
           %lid1 = OpIAdd %ulong %ulong_1 %lid_x
              %v = OpIAdd %ulong %m2 %ulong_max
                 OpBranch %merge
          %merge = OpLabel
              %p = OpPhi %ulong %v %compute %undef %check_n
                 OpControlBarrier %uint_2 %uint_2 %uint_784
         %inb    = OpULessThan %bool %p %ulong_max
                 OpBranchConditional %inb %done %report
         %report = OpLabel
         %printf = OpExtInst %uint %std printf %fmt
                 OpBranch %exit
           %done = OpLabel
                 OpBranch %exit
         %exit_b = OpLabel
                 OpBranch %exit
         %exit_a = OpLabel
                 OpBranch %exit
           %exit = OpLabel
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

  FILE *f = fopen(SRCDIR "/test_remat_alloca_without_initializer.spv", "rb");
  if (!f) {
    printf("Failed to open test_remat_alloca_without_initializer.spv\n");
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

  // n = 2 and m = 1, in a single work-group of 2 work-items, so that both
  // take the path through the barrier, compute v = 1 and print nothing.
  cl_ulong ctx[8] = {2, 0, 0, 0, 1, 0, 0, 0};
  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(ctx), ctx, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(buf), &buf));

  size_t global_size = 2, local_size = 2;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clFinish(queue));

  clReleaseMemObject(buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  printf("OK\n");
  return 0;
}
