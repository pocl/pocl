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

/* A SPIR-V 1.4 kernel that calls through a function pointer, so it declares
   OpCapability FunctionPointersINTEL and OpExtension
   "SPV_INTEL_function_pointers". That extension is not in the set PoCL enables
   for the CPU devices, so the SPIR-V reader rejects the module with
   "input SPIR-V module uses extension 'SPV_INTEL_function_pointers' which were
   disabled by --spirv-ext option". Any module the reader rejects reproduces
   this equally well. */
static const cl_uint UnsupportedExtensionSpirv[] = {
  0x07230203, 0x00010400, 0x0006000e, 0x00000020, 0x00000000, 0x00020011,
  0x00000004, 0x00020011, 0x00000006, 0x00020011, 0x0000000b, 0x00020011,
  0x00000026, 0x00020011, 0x00000027, 0x00020011, 0x000015e3, 0x0008000a,
  0x5f565053, 0x45544e49, 0x75665f4c, 0x6974636e, 0x705f6e6f, 0x746e696f,
  0x00737265, 0x0005000b, 0x00000001, 0x6e65704f, 0x732e4c43, 0x00006474,
  0x0003000e, 0x00000002, 0x00000002, 0x0005000f, 0x00000006, 0x00000015,
  0x0000006b, 0x0000000f, 0x00030010, 0x00000015, 0x0000001f, 0x00030003,
  0x00000000, 0x00000000, 0x00040005, 0x0000000a, 0x6c6c6163, 0x00006565,
  0x00040005, 0x0000000c, 0x6c6c6163, 0x00006565, 0x00030005, 0x0000000f,
  0x00007476, 0x00030005, 0x00000016, 0x0074756f, 0x00030005, 0x0000001a,
  0x00007070, 0x00030005, 0x0000001b, 0x00007066, 0x00030005, 0x0000001d,
  0x00637066, 0x00030005, 0x0000001f, 0x00000072, 0x00030047, 0x0000000f,
  0x00000016, 0x00040015, 0x00000002, 0x00000040, 0x00000000, 0x00040015,
  0x00000004, 0x00000008, 0x00000000, 0x00040015, 0x00000008, 0x00000020,
  0x00000000, 0x0005002b, 0x00000002, 0x00000003, 0x00000001, 0x00000000,
  0x0004002b, 0x00000008, 0x00000011, 0x00000007, 0x0004002b, 0x00000008,
  0x00000018, 0x00000000, 0x00040020, 0x00000005, 0x00000008, 0x00000004,
  0x0004001c, 0x00000006, 0x00000005, 0x00000003, 0x00040020, 0x00000007,
  0x00000005, 0x00000006, 0x00030021, 0x00000009, 0x00000008, 0x00040020,
  0x0000000b, 0x00000007, 0x00000009, 0x00020013, 0x00000012, 0x00040020,
  0x00000013, 0x00000005, 0x00000008, 0x00040021, 0x00000014, 0x00000012,
  0x00000013, 0x00040020, 0x00000019, 0x00000005, 0x00000005, 0x00040020,
  0x0000001c, 0x00000007, 0x00000004, 0x000415e0, 0x0000000b, 0x0000000c,
  0x0000000a, 0x00050034, 0x00000005, 0x0000000d, 0x00000079, 0x0000000c,
  0x0004002c, 0x00000006, 0x0000000e, 0x0000000d, 0x0005003b, 0x00000007,
  0x0000000f, 0x00000005, 0x0000000e, 0x00050036, 0x00000008, 0x0000000a,
  0x00000000, 0x00000009, 0x000200f8, 0x00000010, 0x000200fe, 0x00000011,
  0x00010038, 0x00050036, 0x00000012, 0x00000015, 0x00000000, 0x00000014,
  0x00030037, 0x00000013, 0x00000016, 0x000200f8, 0x00000017, 0x00060043,
  0x00000019, 0x0000001a, 0x0000000f, 0x00000018, 0x00000018, 0x0006003d,
  0x00000005, 0x0000001b, 0x0000001a, 0x00000002, 0x00000008, 0x0004007a,
  0x0000001c, 0x0000001d, 0x0000001b, 0x0004007c, 0x0000000b, 0x0000001e,
  0x0000001d, 0x000415e1, 0x00000008, 0x0000001f, 0x0000001e, 0x0005003e,
  0x00000016, 0x0000001f, 0x00000002, 0x00000004, 0x000100fd, 0x00010038
};

int
main (void)
{
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

  program = clCreateProgramWithIL (ctx, UnsupportedExtensionSpirv,
                                   sizeof (UnsupportedExtensionSpirv), &err);
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
