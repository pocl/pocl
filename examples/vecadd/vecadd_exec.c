#include "poclu.h"
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int
exec_vecadd_kernel (cl_context context, cl_device_id device,
                    cl_command_queue cmd_queue, cl_program program, int n,
                    int wg_size,
                    cl_float *srcA, cl_float *srcB, cl_float *dst)
{
  cl_kernel kernel = NULL;
  cl_mem memobjs[3] = { 0, 0, 0 };
  size_t global_work_size[1];
  size_t local_work_size[1];
  cl_int err = CL_SUCCESS;
  int i;

  poclu_bswap_cl_float_array (device, (cl_float *)srcA, 4 * n);
  poclu_bswap_cl_float_array (device, (cl_float *)srcB, 4 * n);

  memobjs[0]
    = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      sizeof (cl_float) * n, srcA, &err);
  CHECK_CL_ERROR2 (err);

  memobjs[1]
    = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      sizeof (cl_float) * n, srcB, &err);
  CHECK_CL_ERROR2 (err);

  memobjs[2] = clCreateBuffer (context, CL_MEM_READ_WRITE,
			       sizeof (cl_float) * n, NULL, &err);
  CHECK_CL_ERROR2 (err);

  kernel = clCreateKernel (program, "vecadd", NULL);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&memobjs[0]);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void *)&memobjs[1]);
  CHECK_CL_ERROR2 (err);

  err = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void *)&memobjs[2]);
  CHECK_CL_ERROR2 (err);

  global_work_size[0] = n;
  local_work_size[0] = wg_size;

  err = clEnqueueNDRangeKernel (cmd_queue, kernel, 1, NULL, global_work_size,
				local_work_size, 0, NULL, NULL);
  CHECK_CL_ERROR2 (err);

  err = clEnqueueReadBuffer (cmd_queue, memobjs[2], CL_TRUE, 0,
			     n * sizeof (cl_float), dst, 0, NULL, NULL);
  CHECK_CL_ERROR2 (err);

  poclu_bswap_cl_float_array (device, (cl_float *)dst, n);
  poclu_bswap_cl_float_array (device, (cl_float *)srcA, 4 * n);
  poclu_bswap_cl_float_array (device, (cl_float *)srcB, 4 * n);

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
