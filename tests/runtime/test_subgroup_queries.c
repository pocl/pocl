/* Test that the host-side (clGetKernelSubGroupInfo) and device-side
   (get_num_sub_groups etc.) sub-group queries agree, also when a sub-group
   size is required with the intel_reqd_sub_group_size kernel attribute
   (issue #2181).

   Copyright (c) 2026 Tim Besard

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#define REQD_SG_SIZE 32

static const char *SRC_TEMPLATE =
  "%s\n"
  "__kernel void\n"
  "probe (__global uint *num, __global uint *max_size, __global uint *size,\n"
  "       __global uint *id)\n"
  "{\n"
  "  size_t i = get_local_linear_id ();\n"
  "  num[i] = get_num_sub_groups ();\n"
  "  max_size[i] = get_max_sub_group_size ();\n"
  "  size[i] = get_sub_group_size ();\n"
  "  id[i] = get_sub_group_id ();\n"
  "}\n";

static const size_t LOCAL_SIZES[][3] = {
  { 1, 1, 1 }, { 3, 1, 1 }, { 1, 3, 1 },
  { 32, 1, 1 }, { 33, 1, 1 }, { 64, 1, 1 },
};
#define NUM_LOCAL_SIZES (sizeof (LOCAL_SIZES) / sizeof (LOCAL_SIZES[0]))
#define MAX_LOCAL_SIZE 64

static int
check_variant (cl_context ctx, cl_device_id dev, cl_command_queue queue,
               int with_attr)
{
  cl_int err;
  char src[2048];
  char attr[128] = "";
  if (with_attr)
    snprintf (attr, sizeof (attr),
              "__attribute__ ((intel_reqd_sub_group_size (%d)))",
              REQD_SG_SIZE);
  snprintf (src, sizeof (src), SRC_TEMPLATE, attr);

  const char *src_ptr = src;
  cl_program program = clCreateProgramWithSource (ctx, 1, &src_ptr, NULL,
                                                  &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  err = clBuildProgram (program, 1, &dev, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    poclu_show_program_build_log (program);
  CHECK_OPENCL_ERROR_IN ("clBuildProgram");
  cl_kernel kernel = clCreateKernel (program, "probe", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

#ifdef CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL
  size_t compile_sg_size = 0;
  err = clGetKernelSubGroupInfo (kernel, dev,
                                 CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL, 0,
                                 NULL, sizeof (compile_sg_size),
                                 &compile_sg_size, NULL);
  if (err == CL_SUCCESS)
    TEST_ASSERT (compile_sg_size == (with_attr ? REQD_SG_SIZE : 0));
#endif

  cl_mem bufs[4];
  for (unsigned i = 0; i < 4; ++i)
    {
      bufs[i] = clCreateBuffer (ctx, CL_MEM_READ_WRITE,
                                sizeof (cl_uint) * MAX_LOCAL_SIZE, NULL,
                                &err);
      CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
      CHECK_CL_ERROR (clSetKernelArg (kernel, i, sizeof (cl_mem), &bufs[i]));
    }

  size_t max_wg_size = 0;
  CHECK_CL_ERROR (clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                   sizeof (max_wg_size), &max_wg_size, NULL));

  for (size_t l = 0; l < NUM_LOCAL_SIZES; ++l)
    {
      const size_t *ls = LOCAL_SIZES[l];
      const size_t wg_size = ls[0] * ls[1] * ls[2];
      if (wg_size > max_wg_size)
        continue;

      size_t host_max = 0, host_count = 0;
      CHECK_CL_ERROR (clGetKernelSubGroupInfo (
        kernel, dev, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
        sizeof (size_t) * 3, ls, sizeof (host_max), &host_max, NULL));
      CHECK_CL_ERROR (clGetKernelSubGroupInfo (
        kernel, dev, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
        sizeof (size_t) * 3, ls, sizeof (host_count), &host_count, NULL));

      TEST_ASSERT (host_max > 0);
      if (with_attr)
        TEST_ASSERT (host_max == REQD_SG_SIZE);
      /* the count must include a possibly smaller trailing sub-group */
      TEST_ASSERT (host_count == (wg_size + host_max - 1) / host_max);

      CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, 3, NULL, ls, ls,
                                              0, NULL, NULL));

      cl_uint dev_num[MAX_LOCAL_SIZE], dev_max[MAX_LOCAL_SIZE],
        dev_size[MAX_LOCAL_SIZE], dev_id[MAX_LOCAL_SIZE];
      cl_uint *results[4] = { dev_num, dev_max, dev_size, dev_id };
      for (unsigned i = 0; i < 4; ++i)
        CHECK_CL_ERROR (clEnqueueReadBuffer (queue, bufs[i], CL_TRUE, 0,
                                             sizeof (cl_uint) * wg_size,
                                             results[i], 0, NULL, NULL));

      for (size_t i = 0; i < wg_size; ++i)
        {
          /* the device-side queries must agree with the host */
          TEST_ASSERT (dev_num[i] == host_count);
          TEST_ASSERT (dev_max[i] == host_max);

          /* every work-item must see the actual size of its sub-group,
             where the last sub-group may be smaller than the maximum */
          size_t sg = i / host_max;
          size_t expected_size = wg_size - sg * host_max;
          if (expected_size > host_max)
            expected_size = host_max;
          TEST_ASSERT (dev_id[i] == sg);
          TEST_ASSERT (dev_size[i] == expected_size);
        }
    }

  for (unsigned i = 0; i < 4; ++i)
    CHECK_CL_ERROR (clReleaseMemObject (bufs[i]));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  return CL_SUCCESS;
}

int
main (int argc, char **argv)
{
  cl_context ctx;
  cl_device_id dev;
  cl_command_queue queue;
  cl_platform_id platform;

  CHECK_CL_ERROR (poclu_get_any_device2 (&ctx, &dev, &queue, &platform));

  cl_uint max_num_sub_groups = 0;
  clGetDeviceInfo (dev, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                   sizeof (max_num_sub_groups), &max_num_sub_groups, NULL);
  if (max_num_sub_groups == 0)
    {
      printf ("SKIP: device does not support sub-groups\n");
      return 77;
    }

  CHECK_CL_ERROR (check_variant (ctx, dev, queue, 0));

  /* only test intel_reqd_sub_group_size where supported */
  char extensions[4096];
  CHECK_CL_ERROR (clGetDeviceInfo (dev, CL_DEVICE_EXTENSIONS,
                                   sizeof (extensions), extensions, NULL));
  if (strstr (extensions, "cl_intel_required_subgroup_size") != NULL)
    CHECK_CL_ERROR (check_variant (ctx, dev, queue, 1));

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
