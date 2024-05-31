#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"
#include "config.h"

static const char *empty_src = "\n";

int main(int argc, char **argv)
{
  cl_int err;
  const char *krn_src;
  cl_program empty, program;
  cl_platform_id pid = NULL;
  cl_context ctx = NULL;
  cl_device_id did = NULL;
  cl_command_queue queue = NULL;
  cl_uint num_krn = 0;
  cl_kernel kernels[2];

  err = poclu_get_any_device2 (&ctx, &did, &queue, &pid);
  CHECK_OPENCL_ERROR_IN("poclu_get_any_device");
  TEST_ASSERT( ctx );
  TEST_ASSERT( did );
  TEST_ASSERT( queue );

  /* Test creating a program from an empty source */
  empty = clCreateProgramWithSource(ctx, 1, &empty_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");
  err = clBuildProgram(empty, 0, NULL, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");

  err = clCreateKernelsInProgram(empty, 0, NULL, &num_krn);
  CHECK_OPENCL_ERROR_IN("clCreateKernelsInProgram");
  TEST_ASSERT(num_krn == 0);

  krn_src = poclu_read_file(SRCDIR "/tests/runtime/test_clCreateKernelsInProgram.cl");
  TEST_ASSERT(krn_src);

  program = clCreateProgramWithSource(ctx, 1, &krn_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");

  err = clCreateKernelsInProgram(program, 0, NULL, &num_krn);
  CHECK_OPENCL_ERROR_IN("clCreateKernelsInProgram");
  // test_clCreateKernelsInProgram.cl has two kernel functions.
  TEST_ASSERT(num_krn == 2);

  err = clCreateKernelsInProgram(program, 2, kernels, NULL);
  CHECK_OPENCL_ERROR_IN("clCreateKernelsInProgram");

  // make sure the kernels were actually created 
  // Note: nothing in the specification says which kernel function
  // is kernels[0], which is kernels[1]. For now assume pocl/LLVM
  // orders these deterministacally
  err = clEnqueueTask(queue, kernels[0], 0, NULL, NULL); 
  CHECK_OPENCL_ERROR_IN("clEnqueueTask");

  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish");

  err = clEnqueueTask(queue, kernels[1], 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueTask");

  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish");

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseKernel (kernels[0]));
  CHECK_CL_ERROR (clReleaseKernel (kernels[1]));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseProgram (empty));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (pid));

  free ((void *)krn_src);

  return EXIT_SUCCESS;
}


