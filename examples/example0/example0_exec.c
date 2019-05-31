#include "poclu.h"
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    poclu_bswap_cl_int_array (device, (cl_uint *)srcA, n);
    poclu_bswap_cl_int_array (device, (cl_uint *)srcB, n);

    memobjs[0] = clCreateBuffer (context, CL_MEM_READ_ONLY,
                                 sizeof (cl_uint) * n, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memobjs[1] = clCreateBuffer (context, CL_MEM_READ_ONLY,
                                 sizeof (cl_uint) * n, NULL, &err);
    CHECK_CL_ERROR2 (err);

    memobjs[2] = clCreateBuffer (context, CL_MEM_READ_WRITE,
                                 sizeof (cl_uint) * n, NULL, &err);
    CHECK_CL_ERROR2 (err);

    err = clEnqueueWriteBuffer (cmd_queue, memobjs[0], CL_TRUE, 0,
                                n * sizeof (cl_uint), srcA, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);

    err = clEnqueueWriteBuffer (cmd_queue, memobjs[1], CL_TRUE, 0,
                                n * sizeof (cl_uint), srcB, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);

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

    err = clEnqueueReadBuffer (cmd_queue, memobjs[2], CL_TRUE, 0,
                               n * sizeof (cl_uint), dst, 0, NULL, NULL);
    CHECK_CL_ERROR2 (err);

    poclu_bswap_cl_int_array (device, (cl_uint *)dst, n);
    poclu_bswap_cl_int_array (device, (cl_uint *)srcA, n);
    poclu_bswap_cl_int_array (device, (cl_uint *)srcB, n);

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
