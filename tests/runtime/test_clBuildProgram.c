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

static const char warning_kernel[] =
  "kernel void test_kernel(int j, k) { return; }\n";

static const char missing_symbol_kernel[] = "kernel void test_kernel() { "
                                            "one_does_not_simply_walk_into_"
                                            "mordor(); }\n";

/* kernel can have any name, except main() starting from OpenCL 2.0 */
static const char valid_kernel[] =
  "kernel void init(global int *arg) { return; }\n";

static const char invalid_build_option[] =
  "-fnothing-to-see-here";

#define FAKE_PTR 0xDEADBEEF

void buildprogram_callback(cl_program program, void *user_data)
{
  fprintf(stderr, "cl_program callback (via pfn_notify)\n");

  if (user_data == (void*)FAKE_PTR)
    fprintf (stderr, "build callback successful\n");
  else
    fprintf (stderr, "build callback FAILED\n");
}


int
main(void){
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES + 1]; // + 1 for duplicate test
  cl_uint num_devices;
  cl_uint i;
  cl_program program = NULL;
  CHECK_CL_ERROR(clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms));
  TEST_ASSERT(nplatforms > 0);

  CHECK_CL_ERROR(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                       devices, &num_devices));

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  /* TEST 1: Dummy kernel includes another kernel */
  {
      size_t kernel_size = strlen(kernel);
      const char* kernel_buffer = kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                                         &kernel_size, &err);
      //clCreateProgramWithSource for the kernel with #include failed
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices,
         "-D__FUNC__=helper_func -I./test_data", NULL, NULL));

      CHECK_CL_ERROR(clReleaseProgram(program));
  }
  /* TEST 2: invalid kernel */
  {
      size_t kernel_size = strlen(invalid_kernel);
      const char* kernel_buffer = invalid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                          &kernel_size, &err);
      //clCreateProgramWithSource for invalid kernel failed
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
      TEST_ASSERT(err == CL_BUILD_PROGRAM_FAILURE);

      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /* TEST 3: preprocess fail kernel with invalid build option */
  {
      size_t kernel_size = strlen(preprocess_fail);
      const char* kernel_buffer = preprocess_fail;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                          &kernel_size, &err);
      //clCreateProgramWithSource for invalid kernel failed
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      err = clBuildProgram(program, num_devices, devices, invalid_build_option, NULL, NULL);
      TEST_ASSERT(err == CL_INVALID_BUILD_OPTIONS);

      for (i = 0; i < num_devices; ++i) {
              size_t log_size = 0;
              CHECK_CL_ERROR(clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                      0, NULL, &log_size));
              char *log = malloc(log_size);
              CHECK_CL_ERROR(clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                      log_size, log, NULL));
              log[log_size] = '\0';
              fprintf(stderr, "preprocess failure log[%u]: %s\n", i, log);
              free(log);
      }
      /*Lets not release the program as we need it in the next test case*/
      /*CHECK_CL_ERROR(clReleaseProgram(program));*/
  }

  /* TEST 4: preprocess fail kernel with valid build option */
  {
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
          free(log);
      }

      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /* Test the possibility to call a kernel 'init'.
   * Due to the delayed linking in current pocl, this will succeed even if it
   * would fail at link time. Force linking by issuing the kernel once.
   */

  /* TEST 5: valid kernel */
  {
      size_t kernel_size = strlen(valid_kernel);
      const char* kernel_buffer = valid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                          &kernel_size, &err);
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, NULL, NULL, NULL));

      /* TODO FIXME: from here to the clFinish() should be removed once
       * delayed linking is disabled/removed in pocl, probably
       */
      cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
      cl_kernel k = clCreateKernel(program, "init", &err);
      CHECK_OPENCL_ERROR_IN("clCreateKernel");

      CHECK_CL_ERROR(clSetKernelArg(k, 0, sizeof(cl_mem), NULL));
      size_t gws[] = {1};
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL));
      CHECK_CL_ERROR(clFinish(q));

      CHECK_CL_ERROR(clReleaseCommandQueue(q));
      CHECK_CL_ERROR(clReleaseKernel(k));
      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /* TEST 6: valid kernel build pfn-nofity */
  {
      size_t  kernel_size = strlen(valid_kernel);
      const char* kernel_buffer = valid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                                          &kernel_size, &err);
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      /*pfn_notify function should print out "Test6" (userdata)*/
      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, NULL, &buildprogram_callback, (void*)FAKE_PTR));

      /* TODO FIXME: from here to the clFinish() should be removed once
       * delayed linking is disabled/removed in pocl, probably
       */
      cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
      cl_kernel k = clCreateKernel(program, "init", &err);
      CHECK_OPENCL_ERROR_IN("clCreateKernel");

      CHECK_CL_ERROR(clSetKernelArg(k, 0, sizeof(cl_mem), NULL));
      size_t gws[] = {1};
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL));
      CHECK_CL_ERROR(clFinish(q));

      CHECK_CL_ERROR(clReleaseCommandQueue(q));
      CHECK_CL_ERROR(clReleaseKernel(k));
      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /* TEST 7: valid kernel with Compile option -cl-strict-aliasing (deprecated after OCL1.0) */
  {
      size_t kernel_size = strlen(valid_kernel);
      const char* kernel_buffer = valid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                                          &kernel_size, &err);
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, "-cl-strict-aliasing", NULL, NULL));

      /* TODO FIXME: from here to the clFinish() should be removed once
       * delayed linking is disabled/removed in pocl, probably
       */
      cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
      cl_kernel k = clCreateKernel(program, "init", &err);
      CHECK_OPENCL_ERROR_IN("clCreateKernel");

      CHECK_CL_ERROR(clSetKernelArg(k, 0, sizeof(cl_mem), NULL));
      size_t gws[] = {1};
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL));
      CHECK_CL_ERROR(clFinish(q));

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
          /*As this build option deprecated after OCL1.0 we should see a warning here*/
          fprintf(stderr, "Deprecated -cl-strict-aliasing log[%u]: %s\n", i, log);

          free(log);

          cl_program_binary_type bin_type = 0;
          err = clGetProgramBuildInfo(program, devices[i],
                                      CL_PROGRAM_BINARY_TYPE,
                                      sizeof(bin_type), (void *)&bin_type,
                                      NULL);
          CHECK_OPENCL_ERROR_IN("get program binary type");

          /* cl_program_binary_type */
          switch(bin_type) {
            case CL_PROGRAM_BINARY_TYPE_NONE: /*0x0*/
              fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_NONE\n");
            break;
            case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT: /*0x1*/
              fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT\n");
            break;
            case CL_PROGRAM_BINARY_TYPE_LIBRARY: /*0x2*/
              fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_LIBRARY\n");
            break;
            case CL_PROGRAM_BINARY_TYPE_EXECUTABLE: /*0x4*/
              fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_EXECUTABLE\n");
            break;
          }
      }

      CHECK_CL_ERROR(clReleaseCommandQueue(q));
      CHECK_CL_ERROR(clReleaseKernel(k));
      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /* TEST 8: valid kernel with Compile option -cl-denorms-are-zero*/
  {
      size_t kernel_size = strlen(valid_kernel);
      const char* kernel_buffer = valid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                                          &kernel_size, &err);
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, "-cl-denorms-are-zero", NULL, NULL));

      /* TODO FIXME: from here to the clFinish() should be removed once
       * delayed linking is disabled/removed in pocl, probably
       */
      cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
      cl_kernel k = clCreateKernel(program, "init", &err);
      CHECK_OPENCL_ERROR_IN("clCreateKernel");

      CHECK_CL_ERROR(clSetKernelArg(k, 0, sizeof(cl_mem), NULL));
      size_t gws[] = {1};
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL));
      CHECK_CL_ERROR(clFinish(q));

      CHECK_CL_ERROR(clReleaseCommandQueue(q));
      CHECK_CL_ERROR(clReleaseKernel(k));
      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /*TEST 9: valid kernel with Compile option -cl-no-signed-zeros */
  {
      size_t kernel_size = strlen(valid_kernel);
      const char* kernel_buffer = valid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                                          &kernel_size, &err);
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, "-cl-no-signed-zeros", NULL, NULL));

      /* TODO FIXME: from here to the clFinish() should be removed once
       * delayed linking is disabled/removed in pocl, probably
       */
      cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
      cl_kernel k = clCreateKernel(program, "init", &err);
      CHECK_OPENCL_ERROR_IN("clCreateKernel");

      CHECK_CL_ERROR(clSetKernelArg(k, 0, sizeof(cl_mem), NULL));
      size_t gws[] = {1};
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL));
      CHECK_CL_ERROR(clFinish(q));

      CHECK_CL_ERROR(clReleaseCommandQueue(q));
      CHECK_CL_ERROR(clReleaseKernel(k));
      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /*TEST 10: valid kernel with Compile option -cl-std=CL2.0*/
  {
      size_t kernel_size = strlen(valid_kernel);
      const char* kernel_buffer = valid_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                                          &kernel_size, &err);
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, "-cl-std=CL2.0", NULL, NULL));

      /* TODO FIXME: from here to the clFinish() should be removed once
       * delayed linking is disabled/removed in pocl, probably
       */
      cl_command_queue q = clCreateCommandQueue(context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
      cl_kernel k = clCreateKernel(program, "init", &err);
      CHECK_OPENCL_ERROR_IN("clCreateKernel");

      CHECK_CL_ERROR(clSetKernelArg(k, 0, sizeof(cl_mem), NULL));
      size_t gws[] = {1};
      CHECK_CL_ERROR(clEnqueueNDRangeKernel(q, k, 1, NULL, gws, NULL, 0, NULL, NULL));
      CHECK_CL_ERROR(clFinish(q));

      CHECK_CL_ERROR(clReleaseCommandQueue(q));
      CHECK_CL_ERROR(clReleaseKernel(k));
      CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /*TEST 11: macro test */
  {
    char* macro_kernel = poclu_read_file(SRCDIR "/tests/runtime/test_clBuildProgram_macros.cl" );
    size_t s = strlen(macro_kernel);
    program = clCreateProgramWithSource(context, 1, (const char**)&macro_kernel,
                                        &s, &err);
    CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

    CHECK_CL_ERROR(clBuildProgram(program, num_devices, devices, NULL, NULL, NULL));

    CHECK_CL_ERROR(clReleaseProgram(program));
  }

  /* TEST 12: warning into error */
  {
      size_t kernel_size = strlen(warning_kernel);
      const char* kernel_buffer = warning_kernel;

      program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer,
                          &kernel_size, &err);
      //clCreateProgramWithSource for invalid kernel failed
      CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

      /* This kernel normally build with 1 warning:
       *warning: type specifier missing, defaults to 'int'
       *kernel void test_kernel(int j, k) { return; }
       *                               ^
       *1 warning generated.
       *
       *with -Werror we make this 1 warning into error
       *error: type specifier missing, defaults to 'int'
       *kernel void test_kernel(int j, k) { return; }
       *                               ^
       *1 error generated.
       *
       *If the Error is not generated this test should fail.
      */
      err = clBuildProgram(program, num_devices, devices, "-Werror", NULL, NULL);
      TEST_ASSERT(err == CL_BUILD_PROGRAM_FAILURE);

      CHECK_CL_ERROR(clReleaseProgram(program));
  }

#if !(defined(LLVM_3_6) || defined(LLVM_3_7) ||  defined(LLVM_3_8))
  /* TEST 13: missing symbols: kernel referring nonexistent function */
  {
    size_t kernel_size = strlen (missing_symbol_kernel);
    const char *kernel_buffer = missing_symbol_kernel;

    program = clCreateProgramWithSource (
        context, 1, (const char **)&kernel_buffer, &kernel_size, &err);
    // clCreateProgramWithSource for invalid kernel failed
    CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");

    err = clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);
    TEST_ASSERT (err == CL_BUILD_PROGRAM_FAILURE);

    CHECK_CL_ERROR (clReleaseProgram (program));
  }
#endif

  printf ("OK\n");

  return EXIT_SUCCESS;
}
