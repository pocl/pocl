#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include "poclu.h"
#include "config.h"
#include "pocl_tests.h"


int main(int argc, char **argv)
{
  cl_int err;
  const char *krn_src;
  cl_program program;
  cl_context ctx;
  cl_device_id did;
  cl_command_queue queue;
  cl_uint num_krn;
  cl_kernel kernels[2];

  poclu_get_any_device(&ctx, &did, &queue);
  TEST_ASSERT( ctx );
  TEST_ASSERT( did );
  TEST_ASSERT( queue );

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

  err = clEnqueueTask(queue, kernels[1], 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueTask");

  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish");

  return EXIT_SUCCESS;
}


