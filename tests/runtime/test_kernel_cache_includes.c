#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"
#include "config.h"

char first_include[] =
    "#define PRINT_DEFINE \"This is printf from the first include\\n\"\n"
    "void test_include() { printf(\"A printf from inside a function 1\\n\"); }\n";

char second_include[] =
    "#define PRINT_DEFINE \"This is printf from the second include\\n\"\n"
    "void test_include() { printf(\"A printf from inside a function 2\\n\"); }\n";


int main(int argc, char **argv)
{
  cl_int err;
  const char *krn_src;
  cl_program program, program2;
  cl_kernel kernel, kernel2;

  cl_platform_id pid = NULL;
  cl_context ctx = NULL;
  cl_device_id did = NULL;
  cl_command_queue queue = NULL;

  poclu_get_any_device2 (&ctx, &did, &queue, &pid);
  TEST_ASSERT(ctx);
  TEST_ASSERT(did);
  TEST_ASSERT(queue);

  krn_src = poclu_read_file(SRCDIR "/tests/runtime/test_kernel_cache_includes.cl");
  TEST_ASSERT(krn_src);

  err = poclu_write_file(BUILDDIR "/tests/runtime/test_include.h", first_include,
                         sizeof(first_include)-1);
  TEST_ASSERT(err == 0);

  program = clCreateProgramWithSource(ctx, 1, &krn_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram 1");

  kernel = clCreateKernel(program, "testk", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel 1");

  size_t off[3] = {0,0,0};
  size_t ws[3] = {1,1,1};

  err = clEnqueueNDRangeKernel(queue, kernel, 3, off, ws, ws, 0, NULL, 0);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel 1");

  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish 1");

  /***************************************/

  program2 = clCreateProgramWithSource(ctx, 1, &krn_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource 2");

  err = poclu_write_file(BUILDDIR "/tests/runtime/test_include.h", second_include,
                         sizeof(second_include)-1);
  TEST_ASSERT(err == 0);

  err = clBuildProgram(program2, 0, NULL, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram 2");

  kernel2 = clCreateKernel(program2, "testk", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel 2");

  err = clEnqueueNDRangeKernel(queue, kernel2, 3, off, ws, ws, 0, NULL, 0);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel 2");

  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish 2");

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseKernel (kernel2));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseProgram (program2));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (pid));

  free ((void *)krn_src);

  return EXIT_SUCCESS;
}
