/* matadd - Matrix addition using work-items.

   Copyright (c) 2018 Pekka Jääskeläinen

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

#include "poclu.h"
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int
exec_matadd_kernel (cl_context context, cl_device_id device,
		    cl_command_queue cmd_queue, cl_program program,
		    int n, int m, int o, cl_float *srcA, cl_float *srcB,
		    cl_float *dst)
{
  cl_kernel kernel = NULL;
  cl_mem memobjs[3] = { 0, 0, 0 };
  size_t global_work_size[] = {n, m, o};
  cl_int err = CL_SUCCESS;
  int i;

  poclu_bswap_cl_float_array (device, (cl_float *)srcA,
			      sizeof (cl_float) * n * m * o);
  poclu_bswap_cl_float_array (device, (cl_float *)srcB,
			      sizeof (cl_float) * n * m * o);

  memobjs[0]
    = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      sizeof (cl_float) * n * m * o, srcA, &err);
  CHECK_CL_ERROR2 (err);

  memobjs[1]
    = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      sizeof (cl_float) * n * m * o, srcB, &err);
  CHECK_CL_ERROR2 (err);

  memobjs[2] = clCreateBuffer (context, CL_MEM_READ_WRITE,
			       sizeof (cl_float) * n * m * o, NULL, &err);
  CHECK_CL_ERROR2 (err);

  kernel = clCreateKernel (program, "boxadd", NULL);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&memobjs[0]);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&memobjs[1]);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *)&memobjs[2]);
  CHECK_CL_ERROR2 (err);

  err = clEnqueueNDRangeKernel (cmd_queue, kernel, 3, NULL, global_work_size,
				NULL, 0, NULL, NULL);
  CHECK_CL_ERROR2 (err);

  err = clEnqueueReadBuffer (cmd_queue, memobjs[2], CL_TRUE, 0,
			     n * m * o * sizeof (cl_float), dst, 0, NULL, NULL);
  CHECK_CL_ERROR2 (err);

  poclu_bswap_cl_float_array (device, (cl_float *)dst,
			      sizeof (cl_float) * n * m * o);
  poclu_bswap_cl_float_array (device, (cl_float *)srcA,
			      sizeof (cl_float) * n * m * o);
  poclu_bswap_cl_float_array (device, (cl_float *)srcB,
			      sizeof (cl_float) * n * m * o);

 ERROR:
  clReleaseMemObject (memobjs[0]);
  clReleaseMemObject (memobjs[1]);
  clReleaseMemObject (memobjs[2]);
  clReleaseKernel (kernel);
  return err;
}

#ifdef __cplusplus
}
#endif
