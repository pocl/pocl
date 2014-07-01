#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
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
  cl_uint num_krn;
  cl_kernel kernel;

  poclu_get_any_device(&ctx, &did, &queue);
  assert( ctx );
  assert( did );
  assert( queue );

  krn_src = poclu_read_file(SRCDIR "/tests/runtime/test_clCreateKernelsInProgram.cl");
  assert(krn_src);

  program = clCreateProgramWithSource(ctx, 1, &krn_src, NULL, NULL);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  assert(err == CL_SUCCESS);

  kernel = clCreateKernel(program, NULL, &err);
  assert(err == CL_INVALID_VALUE);

  kernel = clCreateKernel(program, "nonexistent_kernel", &err);
  assert(err == CL_INVALID_KERNEL_NAME);

  printf("OK\n");

  return 0;
}


