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
#include <stdio.h>
#include <stdlib.h>
#include "poclu.h"
#include "config.h"
#include "pocl.h"

char kernelSourceCode[] =
"constant sampler_t samp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"typedef float4 tasty_float;\n"
"kernel \n"
"__attribute__((reqd_work_group_size(256, 1, 1)))"
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

#define SPIR_FILE(NUM, SUFFIX) SPIR_FILE_2(NUM) SUFFIX
#define SPIR_FILE_2(NUM) SRCDIR "/tests/runtime/clGetKernelArgInfo.spir" #NUM

int test_program(cl_program program, int is_spir) {

  cl_int err;
  size_t retsize;
  cl_kernel test_kernel = NULL;
  cl_kernel test_kernel2 = NULL;
  union {
    cl_kernel_arg_address_qualifier address;
    cl_kernel_arg_access_qualifier access;
    cl_kernel_arg_type_qualifier type;
    char string[BUF_LEN];
  } kernel_arg;
  unsigned i;

  test_kernel = clCreateKernel (program, "test_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  test_kernel2 = clCreateKernel (program, "test_kernel2", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  /* ADDR SPACE QUALIFIER tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c
  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  printf ("adr: %x\n", kernel_arg.address);
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_CONSTANT) &&
	      "arg of test_kernel is not CONSTANT addr space");

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_GLOBAL) &&
	      "arg of test_kernel is not GLOBAL addr space");


  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_GLOBAL) &&
	      "arg of test_kernel is not GLOBAL addr space");


  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_PRIVATE) &&
	      "arg of test_kernel is not PRIVATE addr space");


  err = clGetKernelArgInfo(test_kernel, 4, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.address==CL_KERNEL_ARG_ADDRESS_LOCAL) &&
	      "arg of test_kernel is not LOCAL addr space");

  /* ACCESS QUALIFIER tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  for (i = 0; i<5; ++i) {
    err = clGetKernelArgInfo(test_kernel, i, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                              BUF_LEN, &kernel_arg.access, &retsize);
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_NONE) &&
		"arg of test_kernel is not NONE access");
  }

  /* TYPE NAME tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT ((retsize == 6)
               && " arg type name size of test_kernel doesn't fit");
  TEST_ASSERT ((strncmp (kernel_arg.string, "char*", 6) == 0)
               && " arg type name of test_kernel doesn't compare");

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT ((retsize == 7)
               && " arg type name size of test_kernel doesn't fit");
  TEST_ASSERT ((strncmp (kernel_arg.string, "float*", 7) == 0)
               && " arg type name of test_kernel doesn't compare");

  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT ((retsize == 6)
               && " arg type name size of test_kernel doesn't fit");
  TEST_ASSERT ((strncmp (kernel_arg.string, "float", 6) == 0)
               && " arg type name of test_kernel doesn't compare");

  err = clGetKernelArgInfo(test_kernel, 4, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT ((retsize == 5)
               && " arg type name size of test_kernel doesn't fit");
  TEST_ASSERT ((strncmp (kernel_arg.string, "int*", 5) == 0)
               && " arg type name of test_kernel doesn't compare");

  /* TYPE QUALIFIER tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.type==CL_KERNEL_ARG_TYPE_CONST) &&
	      "type qualifier of arg of test_kernel is not CONST");

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.type==CL_KERNEL_ARG_TYPE_VOLATILE) &&
	      "type qualifier of arg of test_kernel is not VOLATILE");

  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_TYPE_QUALIFIER,
                            BUF_LEN, &kernel_arg.type, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.type==CL_KERNEL_ARG_TYPE_NONE) &&
	      "type qualifier of arg of test_kernel is not NONE");

#ifndef LLVM_OLDER_THAN_5_0
  if (!is_spir) {

    /* Clang versions before 5 added the const MD also for non-pointer
       types even though OpenCL specs mandates to put them only in
       pointer args. This was fixed in Clang r299192 (see below).
       TODO: update the SPIRs with 5.0+ Clang to drop the metadata. */

    /*
      r299192 | echuraev | 2017-03-31 13:14:52 +0300 (Fri, 31 Mar 2017) | 26 lines

      [OpenCL] Do not generate "kernel_arg_type_qual" metadata for non-pointer args

      Summary:
      "kernel_arg_type_qual" metadata should contain const/volatile/restrict
      tags only for pointer types to match the corresponding requirement of
      the OpenCL specification.

      OpenCL 2.0 spec 5.9.3 Kernel Object Queries:

      CL_KERNEL_ARG_TYPE_VOLATILE is returned if the argument is a pointer
      and the referenced type is declared with the volatile qualifier.
      [...]
      Similarly, CL_KERNEL_ARG_TYPE_CONST is returned if the argument is a
      pointer and the referenced type is declared with the restrict or const
      qualifier.
      [...]
      CL_KERNEL_ARG_TYPE_RESTRICT will be returned if the pointer type is
      marked restrict.
    */
    err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_TYPE_QUALIFIER,
                             BUF_LEN, &kernel_arg.type, &retsize);
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT((kernel_arg.type == CL_KERNEL_ARG_TYPE_NONE));
  }
#endif

  /* NAME tests */
  // constant char* msg, global volatile float* in, global float* out, const float j, local int* c

  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  if (is_spir && (err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)) {
    printf("arg name not available (this is normal for SPIR)\n");
  } else {
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT ((retsize == 4)
                 && " arg name size of test_kernel doesn't fit");
    TEST_ASSERT ((strncmp (kernel_arg.string, "msg", 4) == 0)
                 && " arg name of test_kernel doesn't compare");
  }

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  if (is_spir && (err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)) {
    printf("arg name not available (this is normal for SPIR)\n");
  } else {
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT ((retsize == 3)
                 && " arg name size of test_kernel doesn't fit");
    TEST_ASSERT ((strncmp (kernel_arg.string, "in", 3) == 0)
                 && " arg name of test_kernel doesn't compare");
  }

  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  if (is_spir && (err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)) {
    printf("arg name not available (this is normal for SPIR)\n");
  } else {
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT ((retsize == 4)
                 && " arg name size of test_kernel doesn't fit");
    TEST_ASSERT ((strncmp (kernel_arg.string, "out", 4) == 0)
                 && " arg name of test_kernel doesn't compare");
  }

  err = clGetKernelArgInfo(test_kernel, 3, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  if (is_spir && (err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)) {
    printf("arg name not available (this is normal for SPIR)\n");
  } else {
    CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
    TEST_ASSERT ((retsize == 2)
                 && " arg name size of test_kernel doesn't fit");
    TEST_ASSERT ((strncmp (kernel_arg.string, "j", 2) == 0)
                 && " arg name of test_kernel doesn't compare");
  }

  /* ACCESS QUALIFIER tests for test_kernel2 */
  // const float4 vec1, __read_only image2d_t in, write_only image2d_t out
  err = clGetKernelArgInfo(test_kernel2, 0, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_NONE) && "arg of test_kernel2 is not NONE access");

  err = clGetKernelArgInfo(test_kernel2, 1, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_READ_ONLY) && "arg of test_kernel2 is not READ_ONLY access");

  err = clGetKernelArgInfo(test_kernel2, 2, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT((kernel_arg.access==CL_KERNEL_ARG_ACCESS_WRITE_ONLY) && "arg of test_kernel2 is not WRITE_ONLY access");

  /* check typedef-ed arg type name */

  err = clGetKernelArgInfo(test_kernel2, 0, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  CHECK_OPENCL_ERROR_IN("clGetKernelArgInfo");
  TEST_ASSERT ((retsize == 12)
               && " arg type name size of test_kernel doesn't fit");
  TEST_ASSERT ((strncmp (kernel_arg.string, "tasty_float", 12) == 0)
               && " arg type name of test_kernel2 doesn't compare");

  err = clReleaseKernel(test_kernel);
  CHECK_OPENCL_ERROR_IN("clReleaseKernel");
  err = clReleaseKernel(test_kernel2);
  CHECK_OPENCL_ERROR_IN("clReleaseKernel");
  return EXIT_SUCCESS;

}

/*
int test_program_nometa(cl_program program) {

  cl_int err;
  size_t retsize;
  cl_kernel test_kernel = NULL;
  union {
    cl_kernel_arg_address_qualifier address;
    cl_kernel_arg_access_qualifier access;
    cl_kernel_arg_type_qualifier type;
    char string[BUF_LEN];
  } kernel_arg;

  test_kernel = clCreateKernel (program, "test_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");


  err = clGetKernelArgInfo(test_kernel, 0, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.address, &retsize);
  TEST_ASSERT(err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                            BUF_LEN, &kernel_arg.access, &retsize);
  TEST_ASSERT(err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

  err = clGetKernelArgInfo(test_kernel, 2, CL_KERNEL_ARG_TYPE_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  TEST_ASSERT(err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

  err = clGetKernelArgInfo(test_kernel, 1, CL_KERNEL_ARG_NAME,
                            BUF_LEN, &kernel_arg.string, &retsize);
  TEST_ASSERT(err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

  err = clReleaseKernel (test_kernel);
  CHECK_OPENCL_ERROR_IN ("clReleaseKernel");

  return EXIT_SUCCESS;
}
*/

int spir_program(char * filename, cl_context ctx, cl_device_id did, cl_program* program) {
  cl_int err;

  size_t program_size;
  char* program_buffer;

  program_buffer = poclu_read_binfile(filename, &program_size);
  TEST_ASSERT(program_size > 2000);
  TEST_ASSERT(program_buffer != NULL);
  printf("program size: %zi\n", program_size);

  *program = clCreateProgramWithBinary (ctx, 1, &did, &program_size,
                                       (const unsigned char**)&program_buffer,
                                        NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithBinary");
  TEST_ASSERT(program);

  CHECK_CL_ERROR(clBuildProgram (*program, 1, &did, NULL, NULL, NULL));

  free (program_buffer);

  return EXIT_SUCCESS;
}

int main()
{
  cl_int err;

  cl_context ctx;
  cl_device_id did;
  cl_command_queue queue;

  size_t program_size;
  char* program_buffer;

  cl_program program = NULL;

  poclu_get_any_device(&ctx, &did, &queue);
  TEST_ASSERT(ctx);
  TEST_ASSERT(did);
  TEST_ASSERT(queue);

  /* regular non-SPIR program */

  program_size = strlen (kernelSourceCode);
  program_buffer = kernelSourceCode;

  program = clCreateProgramWithSource (ctx, 1,
                                       (const char**)&program_buffer,
                                       &program_size, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithBinary");
  TEST_ASSERT(program);

  err = clBuildProgram (program, 1, &did, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");

  printf("\nNON-SPIR\n");
  TEST_ASSERT(test_program(program, 0) == EXIT_SUCCESS);

  CHECK_CL_ERROR(clReleaseProgram(program));

  char extensions[1024];
  err = clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, 1024, extensions, NULL);
  CHECK_OPENCL_ERROR_IN("clGetDeviceInfo");
  if (strstr(extensions, "cl_khr_spir") == NULL)
    {
      printf ("SPIR not supported, skipping SPIR arg info tests\n");
      goto FINISH;
    }

  /* SPIR program */

  printf("\nSPIR with metadata\n");
  TEST_ASSERT(spir_program(SPIR_FILE(POCL_DEVICE_ADDRESS_BITS, "_meta"), ctx, did, &program) == EXIT_SUCCESS);

  TEST_ASSERT(test_program(program, 1) == EXIT_SUCCESS);

  CHECK_CL_ERROR(clReleaseProgram(program));

  /* SPIR program without metadata - currently disabled since Clang seems to
   * always generate metadata. */
/*
  printf("\nSPIR WITHOUT metadata\n");
  TEST_ASSERT(spir_program(SPIR_FILE(POCL_DEVICE_ADDRESS_BITS, "_nometa"), ctx, did, &program) == EXIT_SUCCESS);

  TEST_ASSERT(test_program_nometa(program) == EXIT_SUCCESS);

  CHECK_CL_ERROR(clReleaseProgram(program));
*/

FINISH:
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadCompiler ());

  printf("\nOK\n");
  return EXIT_SUCCESS;

}
