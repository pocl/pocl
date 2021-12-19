#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#ifdef _WIN32
#  include "vccompat.hpp"
#endif

#ifndef SRCDIR
#  define SRCDIR="."
#endif

int call_test(const char *name)
{
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  size_t srcdir_length, name_length, filename_size;
  char *filename = NULL;
  char *source = NULL;
  cl_device_id devices[1];
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_int result;
  int retval = -1;

  TEST_ASSERT (name != NULL);

  /* determine file name of kernel source to load */
  srcdir_length = strlen(SRCDIR);
  name_length = strlen(name);
  filename_size = srcdir_length + name_length + 16;
  filename = (char *)malloc(filename_size + 1);
  if (!filename) {
    puts("out of memory");
    goto error;
  }

  snprintf(filename, filename_size, "%s/%s.cl", SRCDIR, name);

  /* read source code */
  source = poclu_read_file (filename);
  TEST_ASSERT (source != NULL && "Kernel .cl not found.");

  /* setup an OpenCL context and command queue using default device */
  context = poclu_create_any_context();
  if (!context) {
    puts("clCreateContextFromType call failed\n");
    goto error;
  }

  result = clGetContextInfo(context, CL_CONTEXT_DEVICES,
      sizeof(cl_device_id), devices, NULL);
  if (result != CL_SUCCESS) {
    puts("clGetContextInfo call failed\n");
    goto error;
  }

  queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queue) {
    puts("clCreateCommandQueue call failed\n");
    goto error;
  }

  /* create and build program */
  program = clCreateProgramWithSource (context, 1, (const char **)&source,
                                       NULL, NULL);
  if (!program) {
    puts("clCreateProgramWithSource call failed\n");
    goto error;
  }

  result = clBuildProgram(program, 0, NULL, "-I" SRCDIR, NULL, NULL);
  if (result != CL_SUCCESS) {
    puts("clBuildProgram call failed\n");
    goto error;
  }

  /* execute the kernel with give name */
  kernel = clCreateKernel(program, name, NULL); 
  if (!kernel) {
    puts("clCreateKernel call failed\n");
    goto error;
  }

  result = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
      global_work_size, local_work_size, 0, NULL, NULL); 
  if (result != CL_SUCCESS) {
    puts("clEnqueueNDRangeKernel call failed\n");
    goto error;
  }

  result = clFinish(queue);
  if (result == CL_SUCCESS)
    retval = 0;

error:

  if (kernel) {
    clReleaseKernel(kernel);
  }
  if (program) {
    clReleaseProgram(program);
  }
  if (queue) {
    clReleaseCommandQueue(queue);
  }
  if (context) {
    clReleaseContext (context);
    clUnloadCompiler ();
  }
  if (source) {
    free(source);
  }
  if (filename) {
    free(filename);
  }

  return retval;
}

const char * const all_tests[] = {
  "test_sizeof",
  "test_as_type",
  "test_convert_type",
  "test_bitselect",
  "test_fabs",
  "test_hadd",
  "test_rotate",
  "test_block",
};
   
const int num_all_tests = (int)(sizeof(all_tests) / sizeof(all_tests[0]));

int main(int argc, char **argv)
{
  int i, retval;

  if (argc < 2) {
    /* Run all tests */
    for (i = 0; i < num_all_tests; ++i) {
      printf("Running test #%d %s...\n", i, all_tests[i]);
      fflush(stdout);
      fflush(stderr);
      retval = call_test(all_tests[i]);
      fflush(stdout);
      fflush(stderr);
    }
  } else {
    /* Run one test */
    printf("Running test %s...\n", argv[1]);
    fflush(stdout);
    fflush(stderr);
    retval = call_test(argv[1]);
    fflush(stdout);
    fflush(stderr);
  }

  CHECK_CL_ERROR (clUnloadCompiler ());

  if (retval)
    printf("FAIL\n");
  else
    printf("OK\n");
  fflush(stdout);
  fflush(stderr);
  return (retval ? 1 : 0);
}

