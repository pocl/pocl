/* example2a - Matrix transpose example from OpenCL specification (using
               automatic locals)

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pocl_opencl.h"

#ifdef _WIN32
#  include "vccompat.hpp"
#endif

#define WIDTH 256
#define HEIGHT 4096
#define PADDING 32

int
main (int argc, char **argv)
{
  cl_float *input = NULL, *output = NULL;
  int i, j, err, spir, spirv, poclbin;
  cl_mem memobjs[2] = { 0 };
  size_t global_work_size[2] = { 0 };
  size_t local_work_size[2] = { 0 };

  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform = NULL;

  err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");

  spir = (argc > 1 && argv[1][0] == 's');
  spirv = (argc > 1 && argv[1][0] == 'v');
  poclbin = (argc > 1 && argv[1][0] == 'b');
  const char *explicit_binary_path = (argc > 2) ? argv[2] : NULL;

  const char *basename = "example2a";
  err = poclu_load_program (context, device, basename, spir, spirv, poclbin,
                            explicit_binary_path, NULL, &program);

  if (err != CL_SUCCESS)
    goto ERROR;

  input = (cl_float *) malloc (WIDTH * HEIGHT * sizeof (cl_float));
  output = (cl_float *) malloc (WIDTH * (HEIGHT + PADDING) * sizeof (cl_float));

  srand48(0);
  for (i = 0; i < WIDTH; ++i)
    {
      for (j = 0; j < HEIGHT; ++j)
      input[i * HEIGHT + j] = (cl_float)drand48();
      for (j = 0; j < (HEIGHT + PADDING); ++j)
      output[i * (HEIGHT + PADDING) + j] = 0.0f;
    }

  memobjs[0] = clCreateBuffer(context,
			      CL_MEM_READ_WRITE,
			      sizeof(cl_float) * WIDTH * (HEIGHT + PADDING), NULL, NULL);
  CHECK_CL_ERROR2 (err);

  memobjs[1] = clCreateBuffer(context,
			      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			      sizeof(cl_float) * WIDTH * HEIGHT, input, NULL);
  CHECK_CL_ERROR2 (err);

  kernel = clCreateKernel(program, "matrix_transpose", NULL);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg(kernel,  0,  
		       sizeof(cl_mem), (void *) &memobjs[0]);
  CHECK_CL_ERROR2 (err);
  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&memobjs[1]);
  CHECK_CL_ERROR2 (err);

  global_work_size[0] = 2 * WIDTH; 
  global_work_size[1] = HEIGHT / 32; 
  local_work_size[0]= 64; 
  local_work_size[1]= 1;

  err = clEnqueueNDRangeKernel (queue, kernel, 2, NULL, global_work_size,
                                local_work_size, 0, NULL, NULL);
  CHECK_CL_ERROR2 (err);

  err = clEnqueueReadBuffer (queue, memobjs[0], CL_TRUE, 0,
                             WIDTH * (HEIGHT + PADDING) * sizeof (cl_float),
                             output, 0, NULL, NULL);
  CHECK_CL_ERROR2 (err);

  err = clFinish (queue);
  CHECK_CL_ERROR2 (err);

  for (i = 0; i < HEIGHT; ++i)
    {
      for (j = 0; j < WIDTH; ++j) {
	if (input[i * WIDTH + j] != output[j * (HEIGHT + PADDING) + i]) {
	  printf ("FAIL\n");
          err = 1;
          goto ERROR;
        }
      }
    }

  printf ("OK\n");

ERROR:
  CHECK_CL_ERROR (clReleaseMemObject (memobjs[0]));
  CHECK_CL_ERROR (clReleaseMemObject (memobjs[1]));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  free (input);
  free (output);

  return 0;
}
