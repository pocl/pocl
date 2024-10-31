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
  char *source = NULL;
  cl_platform_id pid = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_int result;
  int retval = -1;

  TEST_ASSERT (name != NULL);

  char filename[1024];
  snprintf (filename, 1023, "kernel/%s", name);

  char Options[1024];
  snprintf (Options, 1024, "-I%s", SRCDIR);

  int err = poclu_get_any_device2 (&context, &device, &queue, &pid);
  CHECK_OPENCL_ERROR_IN ("poclu_get_any_device");

  /* read source code */
  err = poclu_load_program (pid, context, device, filename, 0, 0, NULL,
                            Options, &program);
  CHECK_OPENCL_ERROR_IN ("clCreateProgram call failed\n");

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
    CHECK_CL_ERROR (clReleaseKernel (kernel));
  }
  if (program) {
    CHECK_CL_ERROR (clReleaseProgram (program));
  }
  if (queue) {
    CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  }
  if (context) {
    CHECK_CL_ERROR (clReleaseContext (context));
    CHECK_CL_ERROR (clUnloadPlatformCompiler (pid));
  }
  if (source) {
    free(source);
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

  if (retval)
    printf("FAIL\n");
  else
    printf("OK\n");
  fflush(stdout);
  fflush(stderr);
  return (retval ? EXIT_FAILURE : EXIT_SUCCESS);
}
