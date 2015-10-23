/* Tests clBuildProgram, passing the user options etc.

   Copyright (c) 2013 Pekka Jääskeläinen and
                      Kalray

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <poclu.h>
#include "config.h"
#include "pocl_tests.h"

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32
#define MAX_BINARIES  32

/* A dummy kernel that just includes another kernel. To test the #include and
   -I */
static const char kernel[] =
  "#include \"test_kernel_src_in_another_dir.h\"\n"
  "#include \"test_kernel_src_in_pwd.h\"\n";

/* A program that fails at preprocess time due to missing endquote
 * in an #include directive
 */
static const char preprocess_fail[] =
  "#include \"missing_endquote.h\n";

static const char invalid_kernel[] =
  "kernel void test_kernel(constant int a, j) { return 3; }\n";


/* kernel can have any name, except main() starting from OpenCL 2.0 */
static const char valid_kernel[] =
  "kernel void init(global int *arg) { return; }\n";

static const char invalid_build_option[] =
  "-fnothing-to-see-here";

int
main(void){
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES + 1]; // + 1 for duplicate test
  cl_uint num_devices;
  cl_uint i;
  cl_program program = NULL;
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;

  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
		       devices, &num_devices);  
  CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  size_t kernel_size = strlen(kernel);
  const char* kernel_buffer = kernel;

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer, 
                                     &kernel_size, &err);
  //clCreateProgramWithSource for the kernel with #include failed
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, num_devices, devices, 
     "-D__FUNC__=helper_func -I./test_data", NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");

  err = clReleaseProgram(program);
  CHECK_OPENCL_ERROR_IN("clReleaseProgram");

  kernel_size = strlen(invalid_kernel);
  kernel_buffer = invalid_kernel;

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
				      &kernel_size, &err);
  //clCreateProgramWithSource for invalid kernel failed
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  TEST_ASSERT(err == CL_BUILD_PROGRAM_FAILURE);

  err = clReleaseProgram(program);
  CHECK_OPENCL_ERROR_IN("clReleaseProgram");

  kernel_size = strlen(preprocess_fail);
  kernel_buffer = preprocess_fail;

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
				      &kernel_size, &err);
  //clCreateProgramWithSource for invalid kernel failed
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, num_devices, devices, invalid_build_option, NULL, NULL);
  TEST_ASSERT(err == CL_INVALID_BUILD_OPTIONS);

  for (i = 0; i < num_devices; ++i) {
          size_t log_size = 0;
          err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                  0, NULL, &log_size);
          CHECK_OPENCL_ERROR_IN("get build log size");
          char *log = malloc(log_size);
          err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                  log_size, log, NULL);
          CHECK_OPENCL_ERROR_IN("get build log");
          log[log_size] = '\0';
          fprintf(stderr, "preprocess failure log[%u]: %s\n", i, log);
  }

  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  TEST_ASSERT(err == CL_BUILD_PROGRAM_FAILURE);

  for (i = 0; i < num_devices; ++i) {
	  size_t log_size = 0;
	  err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
		  0, NULL, &log_size);
	  CHECK_OPENCL_ERROR_IN("get build log size");
	  char *log = malloc(log_size);
	  err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
		  log_size, log, NULL);
	  CHECK_OPENCL_ERROR_IN("get build log");
	  log[log_size] = '\0';
	  fprintf(stderr, "preprocess failure log[%u]: %s\n", i, log);
  }

  err = clReleaseProgram(program);
  CHECK_OPENCL_ERROR_IN("clReleaseProgram");

  /* Test the possibility to call a kernel 'init'.
   * Due to the delayed linking in current pocl, this will succeed even if it
   * would fail at link time. Force linking by issuing the kernel once.
   */

  kernel_size = strlen(valid_kernel);
  kernel_buffer = valid_kernel;

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
				      &kernel_size, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  TEST_ASSERT(err == CL_SUCCESS);

  /* TODO FIXME: from here to the clFinish() should be removed once
   * delayed linking is disabled/removed in pocl, probably
   */
  cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
  cl_kernel k = clCreateKernel(program, "init", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  err = clSetKernelArg(k, 0, sizeof(cl_mem), NULL);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg");
  size_t gws[] = {1};
  err = clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
  err = clFinish(q);
  TEST_ASSERT(err == CL_SUCCESS);

  err  = clReleaseCommandQueue(q);
  err |= clReleaseKernel(k);
  err |= clReleaseProgram(program);
  CHECK_OPENCL_ERROR_IN("'init' kernel name test clean-up");

  // macro test
  char* macro_kernel = poclu_read_file(SRCDIR "/tests/runtime/test_clBuildProgram_macros.cl" );
  size_t s = strlen(macro_kernel);
  program = clCreateProgramWithSource(context, 1, (const char**)&macro_kernel,
                                      &s, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  TEST_ASSERT(err == CL_SUCCESS);

  err = clReleaseProgram(program);
  CHECK_OPENCL_ERROR_IN("clReleaseProgram");

  return EXIT_SUCCESS;
}
