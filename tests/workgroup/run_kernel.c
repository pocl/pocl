/* run_kernel - a generic launcher for a kernel without inputs and outputs

   Copyright (c) 2012,2019 Pekka Jääskeläinen

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

#include "poclu.h"
#include <CL/opencl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif

/**
 * The test kernels are assumed to:
 *
 * 1) called 'test_kernel'
 * 2) a) have no inputs or outputs, then only work item id printfs to verify
 *       the correct workgroup transformations, or
 *    b) have a single output buffer with the grid size of integers, which
 *       is dumped after executing the test, to avoid the need for relying
 *       on printf or work-item ordering in the verification
 * 3) executable with any local and global dimensions and sizes
 *
 * Usage:
 *
 * ./run_kernel somekernel.cl 2 2 3 4
 *
 * Where the first integer is the number of work groups to execute and the
 * rest are the local dimensions.
 */
int
main (int argc, char **argv)
{
  char *source;
  cl_context context;
  size_t cb;
  cl_device_id *devices;
  cl_command_queue cmd_queue;
  cl_program program;
  cl_int err;
  cl_kernel kernel;
  size_t global_work_size[3];
  size_t local_work_size[3];
  char kernel_path[2048];

  snprintf (kernel_path, 2048,  "%s/%s", SRCDIR, argv[1]);

  source = poclu_read_file (kernel_path);
  TEST_ASSERT (source != NULL && "Kernel .cl not found.");

  local_work_size[0] = atoi(argv[3]);
  local_work_size[1] = atoi(argv[4]);
  local_work_size[2] = atoi(argv[5]);

  global_work_size[0] = local_work_size[0] * atoi(argv[2]);
  global_work_size[1] = local_work_size[1];
  global_work_size[2] = local_work_size[2];

  context = poclu_create_any_context();
  if (context == (cl_context)0)
    return -1;

  clGetContextInfo (context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
  devices = (cl_device_id *) malloc(cb);
  clGetContextInfo (context, CL_CONTEXT_DEVICES, cb, devices, NULL);

  cmd_queue = clCreateCommandQueue (context, devices[0], 0, NULL);
  if (cmd_queue == (cl_command_queue)0)
    {
      clReleaseContext (context);
      free (devices);
      return -1;
    }
  free (devices);

  program = clCreateProgramWithSource (context, 1, (const char **)&source,
                                       NULL, NULL);
  if (program == (cl_program)0)
    {
      clReleaseCommandQueue (cmd_queue);
      clReleaseContext (context);
      return -1;
    }

  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      clReleaseProgram (program);
      clReleaseCommandQueue (cmd_queue);
      clReleaseContext (context);
      return -1;
    }

  kernel = clCreateKernel (program, "test_kernel", NULL);
  if (kernel == (cl_kernel)0)
    {
      clReleaseProgram (program);
      clReleaseCommandQueue (cmd_queue);
      clReleaseContext (context);
      return -1;
    }

  cl_uint num_args = 0;
  clGetKernelInfo (kernel, CL_KERNEL_NUM_ARGS, sizeof (num_args), &num_args,
                   NULL);

  size_t grid_size
      = global_work_size[0] * global_work_size[1] * global_work_size[2];
  cl_mem outbuf;
  if (num_args == 1)
    {
      cl_int err;
      outbuf = clCreateBuffer (context, CL_MEM_READ_WRITE,
                               grid_size * sizeof (cl_int), NULL, &err);
      if (err != CL_SUCCESS)
        {
          clReleaseKernel (kernel);
          clReleaseProgram (program);
          clReleaseCommandQueue (cmd_queue);
          clReleaseContext (context);
          return -1;
        }
      clSetKernelArg (kernel, 0, sizeof (outbuf), &outbuf);
    }
  else if (num_args != 0)
    assert (num_args == 0 || num_args == 1);

  err = clEnqueueNDRangeKernel (cmd_queue, kernel, 3, NULL, global_work_size,
                                local_work_size, 0, NULL, NULL);
  if (err == CL_SUCCESS && num_args == 1)
    {
      cl_int kern_output[grid_size];
      err = clEnqueueReadBuffer (cmd_queue, outbuf, CL_TRUE, 0,
                                 grid_size * sizeof (cl_int), kern_output, 0,
                                 NULL, NULL);
      for (size_t i = 0; i < grid_size; ++i)
        printf ("%lu: %d\n", i, kern_output[i]);
    }

  if(err != CL_SUCCESS)
    {
      if (num_args == 1)
        clReleaseMemObject (outbuf);
      clReleaseKernel (kernel);
      clReleaseProgram (program);
      clReleaseCommandQueue (cmd_queue);
      clReleaseContext (context);
      return -1;
    }
  clFinish(cmd_queue);

  if (num_args == 1)
    clReleaseMemObject (outbuf);
  clReleaseKernel (kernel);
  clReleaseProgram (program);
  clReleaseCommandQueue (cmd_queue);
  clReleaseContext (context);

  return 0;
}
