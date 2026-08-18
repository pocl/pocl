/* Tests that clBuildProgram reports an error for a SPIR-V module it cannot
   accept, instead of terminating the host process.

   clBuildProgram() reaches llvm::readSpirv(), whose documented contract is to
   return false and fill in an error string. It instead calls std::exit() by way
   of SPIRVErrorLog::checkError(), because SPIRV::SPIRVDbgError defaults to
   SPIRVDbgErrorHandlingKinds::Exit. The host application is killed with no way
   to detect or report the failure, which the OpenCL specification does not
   allow.

   Copyright (c) 2026 PoCL developers

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

#include "config.h"
#include "poclu.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

/* The module is a SPIR-V 1.4 kernel that calls through a function pointer, so
   it declares OpCapability FunctionPointersINTEL and OpExtension
   "SPV_INTEL_function_pointers". That extension is not in the set PoCL enables
   for the CPU devices, so the SPIR-V reader rejects it with "input SPIR-V
   module uses extension 'SPV_INTEL_function_pointers' which were disabled by
   --spirv-ext option". Any module the reader rejects reproduces this equally
   well. */
static const char *DefaultSpirvPath
    = "test_clBuildProgram_unsupported_spirv.spv";

int
main (int argc, char **argv)
{
  const char *input_spirv = (argc > 1) ? argv[1] : DefaultSpirvPath;
  unsigned char *binary;
  size_t binary_size;
  FILE *f;
  cl_context ctx;
  cl_device_id dev;
  cl_command_queue queue;
  cl_platform_id platform;
  cl_program program;
  cl_int err;
  size_t il_version_size = 0;
  size_t log_size = 0;

  err = poclu_get_any_device2 (&ctx, &dev, &queue, &platform);
  CHECK_OPENCL_ERROR_IN ("poclu_get_any_device2");

  err = clGetDeviceInfo (dev, CL_DEVICE_IL_VERSION, 0, NULL, &il_version_size);
  if (err != CL_SUCCESS || il_version_size <= 1)
    {
      printf ("device does not accept SPIR-V, skipping\n");
      return 77;
    }

  f = fopen (input_spirv, "rb");
  if (!f)
    {
      printf ("Failed to open %s at %s:%d\n", input_spirv, __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  fseek (f, 0, SEEK_END);
  binary_size = ftell (f);
  fseek (f, 0, SEEK_SET);
  binary = (unsigned char *)malloc (binary_size);
  TEST_ASSERT (binary != NULL);
  if (fread (binary, 1, binary_size, f) != binary_size)
    {
      printf ("Failed to read %s at %s:%d\n", input_spirv, __FILE__, __LINE__);
      free (binary);
      fclose (f);
      return EXIT_FAILURE;
    }
  fclose (f);

  program = clCreateProgramWithIL (ctx, binary, binary_size, &err);
  free (binary);
  if (err != CL_SUCCESS)
    {
      /* Rejecting the module this early is also a correct outcome. */
      printf ("clCreateProgramWithIL rejected the module with %d\n", err);
      TEST_ASSERT (clReleaseCommandQueue (queue) == CL_SUCCESS);
      TEST_ASSERT (clReleaseContext (ctx) == CL_SUCCESS);
      printf ("OK\n");
      return EXIT_SUCCESS;
    }

  /* The process must survive this call. Before the fix it never returns: the
     SPIR-V reader calls std::exit() from inside it. */
  err = clBuildProgram (program, 1, &dev, NULL, NULL, NULL);
  printf ("clBuildProgram returned %d\n", err);
  TEST_ASSERT (err != CL_SUCCESS);

  err = clGetProgramBuildInfo (program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
                               &log_size);
  CHECK_OPENCL_ERROR_IN ("clGetProgramBuildInfo size");
  if (log_size > 1)
    {
      char *log = (char *)malloc (log_size + 1);
      TEST_ASSERT (log != NULL);
      err = clGetProgramBuildInfo (program, dev, CL_PROGRAM_BUILD_LOG,
                                   log_size, log, NULL);
      CHECK_OPENCL_ERROR_IN ("clGetProgramBuildInfo");
      log[log_size] = 0;
      printf ("build log: %s\n", log);
      free (log);
    }

  TEST_ASSERT (clReleaseProgram (program) == CL_SUCCESS);
  TEST_ASSERT (clReleaseCommandQueue (queue) == CL_SUCCESS);
  TEST_ASSERT (clReleaseContext (ctx) == CL_SUCCESS);
  TEST_ASSERT (clUnloadPlatformCompiler (platform) == CL_SUCCESS);

  printf ("OK\n");
  return EXIT_SUCCESS;
}
