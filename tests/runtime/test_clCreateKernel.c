#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"
#include "config.h"


int main(int argc, char **argv)
{
  cl_int err;
  const char *krn_src;
  cl_program program;
  cl_context ctx;
  cl_command_queue queue;
  cl_device_id did;
  cl_kernel kernel;

  CHECK_CL_ERROR(poclu_get_any_device(&ctx, &did, &queue));
  TEST_ASSERT(ctx);
  TEST_ASSERT(did);
  TEST_ASSERT(queue);

  krn_src = poclu_read_file(SRCDIR "/tests/runtime/test_clCreateKernelsInProgram.cl");
  TEST_ASSERT(krn_src);

  program = clCreateProgramWithSource(ctx, 1, &krn_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  CHECK_CL_ERROR(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

  kernel = clCreateKernel(program, NULL, &err);
  TEST_ASSERT(err == CL_INVALID_VALUE);
  TEST_ASSERT(kernel == NULL);

  kernel = clCreateKernel(program, "nonexistent_kernel", &err);
  TEST_ASSERT(err == CL_INVALID_KERNEL_NAME);
  TEST_ASSERT(kernel == NULL);

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadCompiler ());

  free ((void *)krn_src);

  printf("OK\n");

  return 0;
}


