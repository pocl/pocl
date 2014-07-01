/* Tests clGetKernelArgInfo

   Copyright (c) 2014 Michal Babej

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
#include <CL/cl.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "poclu.h"
#include "pocl_tests.h"

char kernelSourceCode[] =
"constant sampler_t samp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"typedef float4 tasty_float;\n"
"kernel \n"
"void test_kernel2(const tasty_float vec1, __read_only image2d_t in, write_only image2d_t out) {\n"
"    const int2 coord = (get_global_id(0),get_global_id(1));\n"
"    float4 b = read_imagef(in, samp, coord);\n"
"    float4 col = b*vec1;\n"
"    write_imagef(out, coord, col);\n"
"}\n"
"kernel \n"
"void test_kernel(constant char* msg, global volatile float* in, global float* out, const float j, local int* c) {\n"
"    printf(\"%s\", msg);\n"
"    const int i = get_global_id(0);\n"
"    out[i] = in[i] * native_sin(j*c[i%50]);\n"
"}\n";


#define BUF_LEN 2000

int main()
{
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  cl_int err;
  size_t retsize;
  cl_platform_id platforms[1];
  cl_uint nplatforms;
  cl_device_id devices[1]; // + 1 for duplicate test
  cl_uint num_devices;
  cl_program program = NULL;
  cl_kernel test_kernel = NULL;
  cl_kernel test_kernel2 = NULL;
  union {
    cl_kernel_arg_address_qualifier address;
    cl_kernel_arg_access_qualifier access;
    cl_kernel_arg_type_qualifier type;
    char string[BUF_LEN];
  } kernel_arg;
  unsigned i;

  err = clGetPlatformIDs(1, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");
  TEST_ASSERT(nplatforms > 0);

  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1,
                       devices, &num_devices);
  CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL,
                                       NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id), devices, NULL);
  CHECK_OPENCL_ERROR_IN("clGetContextInfo");

  size_t kernel_size = strlen (kernelSourceCode);
  char* kernel_buffer = kernelSourceCode;

  program = clCreateProgramWithSource (context, 1,
                                       (const char**)&kernel_buffer,
                                       &kernel_size, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");

  test_kernel = clCreateKernel (program, "test_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  test_kernel2 = clCreateKernel (program, "test_kernel2", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  /* ADDR SPACE QUALIFIER tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c
  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_CONSTANT) && "arg of test_kernel is not CONSTANT addr space");

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_GLOBAL) && "arg of test_kernel is not GLOBAL addr space");


  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_GLOBAL) && "arg of test_kernel is not GLOBAL addr space");


  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_PRIVATE) && "arg of test_kernel is not PRIVATE addr space");


  err = clGetKernelArgInfo(test_kernel, 4, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_LOCAL) && "arg of test_kernel is not LOCAL addr space");

  /* ACCESS QUALIFIER tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  for (i = 0; i<5; ++i) {
    err = clGetKernelArgInfo(test_kernel, i, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                              BUF_LEN, &kernel_arg.access, &retsize);
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_NONE) && "arg of test_kernel is not NONE access");
  }

  /* TYPE NAME tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==6) && " arg type name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "char*", 6)==0) && " arg type name of test_kernel doesnt compare");


  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==7) && " arg type name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "float*", 7)==0) && " arg type name of test_kernel doesnt compare");

  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==6) && " arg type name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "float", 6)==0) && " arg type name of test_kernel doesnt compare");

  err = clGetKernelArgInfo(test_kernel, 4, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==5) && " arg type name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "int*", 5)==0) && " arg type name of test_kernel doesnt compare");

  /* TYPE QUALIFIER tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_TYPE_CONST) && "type qualifier of arg of test_kernel is not CONST");

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_TYPE_VOLATILE) && "type qualifier of arg of test_kernel is not VOLATILE");

  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_TYPE_NONE) && "type qualifier of arg of test_kernel is not NONE");

  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_TYPE_CONST) && "type qualifier of arg of test_kernel is not CONST");

  /* NAME tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==4) && " arg name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "msg", 4)==0) && " arg name of test_kernel doesnt compare");


  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==3) && " arg name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "in", 3)==0) && " arg name of test_kernel doesnt compare");

  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==4) && " arg name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "out", 4)==0) && " arg name of test_kernel doesnt compare");

  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==2) && " arg name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "j", 2)==0) && " arg name of test_kernel doesnt compare");

  /* ACCESS QUALIFIER tests for test_kernel2 */
  // const float4 vec1, __read_only image2d_t in, write_only image2d_t out
  err = clGetKernelArgInfo(test_kernel2, 0, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_NONE) && "arg of test_kernel2 is not NONE access");

  err = clGetKernelArgInfo(test_kernel2, 1, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_READ_ONLY) && "arg of test_kernel2 is not NONE access");

  err = clGetKernelArgInfo(test_kernel2, 2, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_WRITE_ONLY) && "arg of test_kernel2 is not NONE access");

  /* check typedef-ed arg type name */

  err = clGetKernelArgInfo(test_kernel2, 0, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((retsize==12) && " arg type name size of test_kernel doesnt fit");
  TEST_ASSERT((strncmp(kernel_arg.string, "tasty_float", 12)==0) && " arg type name of test_kernel2 doesnt compare");

  printf("OK\n");
  return EXIT_SUCCESS;

}
