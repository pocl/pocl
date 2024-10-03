/*
  Copyright (c) 2018 Michal Babej / Tampere University

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

#include "config.h"
#include "poclu.h"

/*
  Multi-device migration test. Creates two buffers (in & out),
  enqueues the same kernel with different parameters on every device from 0 to
  N-1, then does the same in opposite direction (N-1 to 0th device). Verifies
  that pocl properly migrates the buffer contents across devices.
*/

#define ITEMS 1024

int
main (int argc, char **argv)
{
  cl_float *input = NULL, *output = NULL;
  int err, total_err, spir, spirv;
  cl_mem buf_in, buf_out;
  size_t global_work_size[2] = { 0 };
  size_t local_work_size[2] = { 0 };

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id *devices = NULL;
  cl_command_queue *queues = NULL;
  cl_uint i, j, num_devices = 0;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_event ev1, ev2;

  err = poclu_get_multiple_devices (&platform, &context, 0, &num_devices,
                                    &devices, &queues,
                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  CHECK_OPENCL_ERROR_IN ("poclu_get_multiple_devices");

  printf ("NUM DEVICES: %u \n", num_devices);
  if (num_devices < 2)
    {
      printf ("NOT ENOUGH DEVICES! (need 2)\n");
      err = 77;
      goto EARLY_EXIT;
    }

  const char *sourcefile = SRCDIR "/tests/runtime/migration_test";
  const char *basename = "migration_test";
  err = poclu_load_program_multidev (context, devices, num_devices, sourcefile,
                                     0, 0, NULL, NULL, &program);
  if (err != CL_SUCCESS)
    goto ERROR;

  char dev_name[1024];
  for (i = 0; i < num_devices; ++i)
    {
      size_t retval;
      err = clGetDeviceInfo (devices[i], CL_DEVICE_NAME, 1024, dev_name,
                             &retval);
      CHECK_CL_ERROR2 (err);

      dev_name[retval] = 0;
      printf ("DEVICE %u is: %s \n", i, dev_name);
    }

  printf ("------------------------\n");

  kernel = clCreateKernel (program, basename, NULL);
  CHECK_CL_ERROR2 (err);

  cl_uint num_floats = num_devices * ITEMS;
  input = (cl_float *)malloc (num_floats * sizeof (cl_float));
  output = (cl_float *)calloc (num_floats, sizeof (cl_float));

  srand (12345);
  for (i = 0; i < num_floats; ++i)
    {
      input[i] = (cl_float)rand ();
    }

  buf_in = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof (cl_float) * num_floats, input, NULL);
  CHECK_CL_ERROR2 (err);

  buf_out = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof (cl_float) * num_floats, output, NULL);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&buf_in);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&buf_out);
  CHECK_CL_ERROR2 (err);

  cl_uint num_items = ITEMS;
  err = clSetKernelArg (kernel, 2, sizeof (cl_uint), &num_items);
  CHECK_CL_ERROR2 (err);

  global_work_size[0] = ITEMS;
  local_work_size[0] = 64;

  fprintf (stderr, "FORWARD \n");

  for (i = 0; i < num_devices; ++i)
    {
      uint32_t index_arg = i;
      fprintf (stderr, "index ARG: %u\n", index_arg);
      err = clSetKernelArg (kernel, 3, sizeof (uint32_t), &index_arg);
      CHECK_CL_ERROR2 (err);

      err = clEnqueueNDRangeKernel (
          queues[i], kernel, 1, NULL, global_work_size, local_work_size,
          (i > 0 ? 1 : 0), (i > 0 ? &ev1 : NULL), &ev2);
      if (i > 0)
        clReleaseEvent (ev1);
      ev1 = ev2;

      CHECK_CL_ERROR2 (err);
    }

  clFinish (queues[num_devices - 1]);

  printf ("------------------------\n");

  fprintf (stderr, "NOW REVERSE \n");

  for (i = num_devices; i > 0; --i)
    {
      uint32_t index_arg = i - 1;
      fprintf (stderr, "index ARG: %u\n", index_arg);
      err = clSetKernelArg (kernel, 3, sizeof (uint32_t), &index_arg);
      CHECK_CL_ERROR2 (err);

      err = clEnqueueNDRangeKernel (queues[i - 1], kernel, 1, NULL,
                                    global_work_size, local_work_size, 1, &ev1,
                                    &ev2);
      clReleaseEvent (ev1);
      ev1 = ev2;

      CHECK_CL_ERROR2 (err);
    }

  err = clEnqueueReadBuffer (queues[0], buf_out, CL_TRUE, 0,
                             num_floats * sizeof (cl_float), output, 1, &ev1,
                             NULL);
  CHECK_CL_ERROR2 (err);
  fprintf (stderr, "DONE \n");

  clReleaseEvent (ev1);

  printf ("------------------------\n");

  fprintf (stderr, "VERIFYING RESULTS \n");

  total_err = 0;
  for (i = 0; i < num_devices; ++i)
    {
      err = 0;
      for (j = 0; j < ITEMS; ++j)
        {
          cl_float actual = output[i * ITEMS + j];
          cl_float expected = input[i * ITEMS + j] * (float)(i + 1) * 2.0f;
          if (expected != actual)
            {
              if (err < 10)
                printf ("FAIL at DEV %u ITEM %u: EXPECTED %e ACTUAL %e\n", i,
                        j, expected, actual);
              err += 1;
              total_err += 1;
            }
        }
      if (err > 0)
        printf ("DEV %u FAILED: %i errs\n", i, err);
      else
        printf ("DEV %u PASS\n", i);
    }
  if (total_err == 0)
    printf ("OK\n");
  else
    printf ("FAIL\n");

ERROR:
  CHECK_CL_ERROR (clReleaseMemObject (buf_in));
  CHECK_CL_ERROR (clReleaseMemObject (buf_out));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));

EARLY_EXIT:
  for (i = 0; i < num_devices; ++i)
    {
      CHECK_CL_ERROR (clReleaseCommandQueue (queues[i]));
    }
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  free (input);
  free (output);
  free (devices);
  free (queues);

  return err;
}
