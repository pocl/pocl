/* example0_exec - helper wrapper for example0

   Copyright (c) 2019 pocl developers

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USE_MAP

#ifdef __cplusplus
extern "C"
{
#endif

  int
  exec_integer_mad_kernel (cl_context context, cl_device_id device,
                           cl_command_queue cmd_queue, cl_program program,
                           unsigned n, cl_uint *srcA, cl_uint *srcB,
                           cl_uint *dst)
  {
    cl_kernel kernel = NULL;
    cl_mem memobjs[3] = { 0, 0, 0 };
    size_t global_work_size[1];
    size_t local_work_size[1];
    cl_int err = CL_SUCCESS;
    int i;
    void *mapped;
    size_t buf_size = sizeof (cl_uint) * n;

    poclu_bswap_cl_int_array (device, (cl_int *)srcA, n);
    poclu_bswap_cl_int_array (device, (cl_int *)srcB, n);

    memobjs[0]
        = clCreateBuffer (context, CL_MEM_READ_ONLY, buf_size, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memobjs[1]
        = clCreateBuffer (context, CL_MEM_READ_ONLY, buf_size, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memobjs[2]
        = clCreateBuffer (context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
    CHECK_CL_ERROR2 (err);

#ifdef USE_MAP

    mapped = clEnqueueMapBuffer (cmd_queue, memobjs[0], CL_TRUE, CL_MAP_WRITE,
                                 0, buf_size, 0, NULL, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memcpy (mapped, srcA, buf_size);

    err = clEnqueueUnmapMemObject (cmd_queue, memobjs[0], mapped, 0, NULL,
                                   NULL);
    CHECK_CL_ERROR2 (err);

    err = clFinish (cmd_queue);
    CHECK_CL_ERROR2 (err);

    mapped = clEnqueueMapBuffer (cmd_queue, memobjs[1], CL_TRUE, CL_MAP_WRITE,
                                 0, buf_size, 0, NULL, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memcpy (mapped, srcB, buf_size);

    err = clEnqueueUnmapMemObject (cmd_queue, memobjs[1], mapped, 0, NULL,
                                   NULL);
    CHECK_CL_ERROR2 (err);

    err = clFinish (cmd_queue);
    CHECK_CL_ERROR2 (err);

#else
    err = clEnqueueWriteBuffer (cmd_queue, memobjs[0], CL_TRUE, 0,
                                n * sizeof (cl_uint), srcA, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);

    err = clEnqueueWriteBuffer (cmd_queue, memobjs[1], CL_TRUE, 0,
                                n * sizeof (cl_uint), srcB, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);
#endif

    kernel = clCreateKernel (program, "integer_mad", NULL);
    CHECK_CL_ERROR2 (err);

    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&memobjs[0]);
    CHECK_CL_ERROR2 (err);

    err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&memobjs[1]);
    CHECK_CL_ERROR2 (err);

    err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *)&memobjs[2]);
    CHECK_CL_ERROR2 (err);

    global_work_size[0] = n;
    local_work_size[0] = 8;

    err = clEnqueueNDRangeKernel (cmd_queue, kernel, 1, NULL, global_work_size,
                                  local_work_size, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);

#ifdef USE_MAP

    mapped = clEnqueueMapBuffer (cmd_queue, memobjs[2], CL_TRUE, CL_MAP_READ,
                                 0, buf_size, 0, NULL, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memcpy (dst, mapped, buf_size);

    err = clEnqueueUnmapMemObject (cmd_queue, memobjs[2], mapped, 0, NULL,
                                   NULL);
    CHECK_CL_ERROR2 (err);

    err = clFinish (cmd_queue);
    CHECK_CL_ERROR2 (err);

#else
    err = clEnqueueReadBuffer (cmd_queue, memobjs[2], CL_TRUE, 0,
                               n * sizeof (cl_uint), dst, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);
#endif

    poclu_bswap_cl_int_array (device, (cl_int *)dst, n);
    poclu_bswap_cl_int_array (device, (cl_int *)srcA, n);
    poclu_bswap_cl_int_array (device, (cl_int *)srcB, n);

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
