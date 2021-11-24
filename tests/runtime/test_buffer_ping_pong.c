/*
  Copyright (c) 2018 Jan Solanti / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#ifdef _MSC_VER
#include "vccompat.hpp"
#endif

/*
  Back-and-forth migration test. Creates one mutable buffer,
  enqueues three kernels that have to be executed in order and reads the result
  back.
*/

char kernelASourceCode[] = "kernel \n"
                           "void write_a(global int* input) {\n"
                           "    input[0] = input[0] + 2;\n"
                           "}\n";

char kernelBSourceCode[] = "kernel \n"
                           "void write_b(global int* input) {\n"
                           "    input[0] = input[0] / 2;\n"
                           "}\n";

char kernelCSourceCode[] = "kernel \n"
                           "void write_c(global int* input) {\n"
                           "    input[0] = input[0] == 5 ? 1 : 0;\n"
                           "}\n";

int
main (int argc, char **argv)
{
  cl_int input = 8, output = 0;
  int err, total_err, spir, spirv;
  cl_mem buf;
  size_t global_work_size = 1;
  size_t local_work_size = 1;

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id *devices = NULL;
  cl_command_queue *queues = NULL;
  cl_uint i, num_devices = 0;
  cl_program program_a = NULL, program_b = NULL, program_c = NULL;
  cl_kernel kernel_a = NULL, kernel_b = NULL, kernel_c = NULL;
  const char *kernel_buffer = NULL;

  err = poclu_get_multiple_devices (&platform, &context, &num_devices,
                                    &devices, &queues);
  CHECK_OPENCL_ERROR_IN ("poclu_get_multiple_devices");

  printf ("NUM DEVICES: %u \n", num_devices);

  kernel_buffer = kernelASourceCode;
  program_a = clCreateProgramWithSource (
      context, 1, (const char **)&kernel_buffer, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource A");
  CHECK_CL_ERROR (clBuildProgram (program_a, 0, NULL, NULL, NULL, NULL));
  kernel_a = clCreateKernel (program_a, "write_a", &err);
  CHECK_CL_ERROR2 (err);

  kernel_buffer = kernelBSourceCode;
  program_b = clCreateProgramWithSource (
      context, 1, (const char **)&kernel_buffer, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource B");
  CHECK_CL_ERROR (clBuildProgram (program_b, 0, NULL, NULL, NULL, NULL));
  kernel_b = clCreateKernel (program_b, "write_b", &err);
  CHECK_CL_ERROR2 (err);

  kernel_buffer = kernelCSourceCode;
  program_c = clCreateProgramWithSource (
      context, 1, (const char **)&kernel_buffer, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource C");
  CHECK_CL_ERROR (clBuildProgram (program_c, 0, NULL, NULL, NULL, NULL));
  kernel_c = clCreateKernel (program_c, "write_c", &err);
  CHECK_CL_ERROR2 (err);

  buf = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        sizeof (cl_int), &input, &err);
  CHECK_CL_ERROR2 (err);

  if (num_devices < 2)
    {
      printf ("NOT ENOUGH DEVICES! (need 2)\n");
      return 77;
    }

  err = clSetKernelArg (kernel_a, 0, sizeof (cl_mem), &buf);
  CHECK_CL_ERROR2 (err);
  err = clSetKernelArg (kernel_b, 0, sizeof (cl_mem), &buf);
  CHECK_CL_ERROR2 (err);
  err = clSetKernelArg (kernel_c, 0, sizeof (cl_mem), &buf);
  CHECK_CL_ERROR2 (err);

  cl_event event_a = NULL, event_b = NULL, event_c = NULL;
  err = clEnqueueNDRangeKernel (queues[0], kernel_a, 1, NULL,
                                &global_work_size, &local_work_size, 0, NULL,
                                &event_a);
  CHECK_CL_ERROR2 (err);
  err = clEnqueueNDRangeKernel (queues[1], kernel_b, 1, NULL,
                                &global_work_size, &local_work_size, 1,
                                &event_a, &event_b);
  CHECK_CL_ERROR2 (err);
  err = clEnqueueNDRangeKernel (queues[0], kernel_c, 1, NULL,
                                &global_work_size, &local_work_size, 1,
                                &event_b, &event_c);
  CHECK_CL_ERROR2 (err);

  err = clEnqueueReadBuffer (queues[0], buf, CL_TRUE, 0, sizeof (cl_int),
                             &output, 1, &event_c, NULL);
  CHECK_CL_ERROR2 (err);
  fprintf (stderr, "DONE \n");

  if (output == 1)
    printf ("OK\n");
  else
    printf ("FAIL\n");

ERROR:
  CHECK_CL_ERROR (clReleaseMemObject (buf));

  for (i = 0; i < num_devices; ++i)
    {
      CHECK_CL_ERROR (clReleaseCommandQueue (queues[i]));
    }

  CHECK_CL_ERROR (clReleaseKernel (kernel_c));
  CHECK_CL_ERROR (clReleaseProgram (program_c));
  CHECK_CL_ERROR (clReleaseKernel (kernel_b));
  CHECK_CL_ERROR (clReleaseProgram (program_b));
  CHECK_CL_ERROR (clReleaseKernel (kernel_a));
  CHECK_CL_ERROR (clReleaseProgram (program_a));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  CHECK_CL_ERROR (clReleaseContext (context));
  free (devices);
  free (queues);

  return err;
}
