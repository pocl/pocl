/* Test that the host-side (clGetKernelSubGroupInfo) and device-side
   (get_num_sub_groups etc.) sub-group queries agree, also when a sub-group
   size is required with the intel_reqd_sub_group_size kernel attribute
   (issue #2181), and that the sub-group count of the largest work-group
   does not exceed CL_DEVICE_MAX_NUM_SUB_GROUPS.

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

/* cl_khr_subgroups' entry point, looked up by name when the device offers
   the extension */
static clGetKernelSubGroupInfoKHR_t *get_info_khr = NULL;

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
      if (get_info_khr != NULL)
        {
          size_t khr_max = 0;
          CHECK_CL_ERROR (get_info_khr (
            kernel, dev, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
            sizeof (size_t) * 3, ls, sizeof (khr_max), &khr_max, NULL));
          TEST_ASSERT (khr_max == host_max);
        }
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

static const char *LARGE_SRC_TEMPLATE = "%s\n"
                                        "__kernel void\n"
                                        "probe (__global uint *num)\n"
                                        "{\n"
                                        "  if (get_local_linear_id () == 0)\n"
                                        "    num[0] = get_num_sub_groups ();\n"
                                        "}\n";

#define SMALL_REQD_SG_SIZE 8

/* Launch the largest work-group the kernel allows, shaped to produce as many
   sub-groups as possible: with a required sub-group size along x, otherwise
   along y, as the default sub-group spans the x dimension. */
static int
check_large_wg (cl_context ctx,
                cl_device_id dev,
                cl_command_queue queue,
                cl_uint dev_max_sgs,
                int with_attr)
{
  cl_int err;
  char src[1024];
  char attr[128] = "";
  if (with_attr)
    snprintf (attr, sizeof (attr),
              "__attribute__ ((intel_reqd_sub_group_size (%d)))",
              SMALL_REQD_SG_SIZE);
  snprintf (src, sizeof (src), LARGE_SRC_TEMPLATE, attr);

  const char *src_ptr = src;
  cl_program program
    = clCreateProgramWithSource (ctx, 1, &src_ptr, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  err = clBuildProgram (program, 1, &dev, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    poclu_show_program_build_log (program);
  CHECK_OPENCL_ERROR_IN ("clBuildProgram");
  cl_kernel kernel = clCreateKernel (program, "probe", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

  cl_mem buf
    = clCreateBuffer (ctx, CL_MEM_READ_WRITE, sizeof (cl_uint), NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  CHECK_CL_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));

  size_t kernel_wg_size = 0;
  CHECK_CL_ERROR (
    clGetKernelWorkGroupInfo (kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
                              sizeof (kernel_wg_size), &kernel_wg_size, NULL));
  size_t item_sizes[3];
  CHECK_CL_ERROR (clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                   sizeof (item_sizes), item_sizes, NULL));
  unsigned dim = with_attr ? 0 : 1;
  size_t wg_size = kernel_wg_size;
  if (wg_size > item_sizes[dim])
    wg_size = item_sizes[dim];
  size_t ls[3] = { 1, 1, 1 };
  ls[dim] = wg_size;

  size_t host_count = 0, kernel_max_sgs = 0;
  CHECK_CL_ERROR (clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, sizeof (ls), ls,
    sizeof (host_count), &host_count, NULL));
  CHECK_CL_ERROR (clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_MAX_NUM_SUB_GROUPS, 0, NULL,
    sizeof (kernel_max_sgs), &kernel_max_sgs, NULL));

  cl_uint dev_num = 0;
  CHECK_CL_ERROR (
    clEnqueueNDRangeKernel (queue, kernel, 3, NULL, ls, ls, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (
    queue, buf, CL_TRUE, 0, sizeof (cl_uint), &dev_num, 0, NULL, NULL));

  TEST_ASSERT (dev_num == host_count);
  TEST_ASSERT (host_count <= kernel_max_sgs);
  TEST_ASSERT (kernel_max_sgs <= dev_max_sgs);
  if (with_attr)
    TEST_ASSERT (kernel_max_sgs
                 == (kernel_wg_size + SMALL_REQD_SG_SIZE - 1)
                      / SMALL_REQD_SG_SIZE);

  /* CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT must return a one-dimensional
     local size that yields exactly the requested count, or zeros. With a
     required size every count up to the largest work-group is reachable;
     the default sub-group spans dimension x, so only a count of 1 is. */
  size_t max_count = with_attr ? wg_size / SMALL_REQD_SG_SIZE : 1;
  size_t wishes[] = { 1, 2, max_count / 2, max_count, max_count + 1 };
  for (size_t w = 0; w < sizeof (wishes) / sizeof (wishes[0]); ++w)
    {
      size_t n = wishes[w];
      if (n == 0)
        continue;
      for (size_t dims = 1; dims <= 3; ++dims)
        {
          size_t nd[3] = { 42, 42, 42 };
          CHECK_CL_ERROR (clGetKernelSubGroupInfo (
            kernel, dev, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof (n),
            &n, sizeof (size_t) * dims, nd, NULL));
          if (n > max_count)
            {
              for (size_t d = 0; d < dims; ++d)
                TEST_ASSERT (nd[d] == 0);
              continue;
            }
          TEST_ASSERT (nd[0] > 0);
          for (size_t d = 1; d < dims; ++d)
            TEST_ASSERT (nd[d] == 1);

          CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, dims, NULL,
                                                  nd, nd, 0, NULL, NULL));
          CHECK_CL_ERROR (clEnqueueReadBuffer (queue, buf, CL_TRUE, 0,
                                               sizeof (cl_uint), &dev_num, 0,
                                               NULL, NULL));
          TEST_ASSERT (dev_num == n);
        }
    }

  CHECK_CL_ERROR (clReleaseMemObject (buf));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  return CL_SUCCESS;
}

static cl_kernel
build_kernel (cl_context ctx, cl_device_id dev, const char *src)
{
  cl_int err;
  cl_program program = clCreateProgramWithSource (ctx, 1, &src, NULL, &err);
  if (err != CL_SUCCESS)
    return NULL;
  err = clBuildProgram (program, 1, &dev, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    poclu_show_program_build_log (program);
  cl_kernel kernel = clCreateKernel (program, "probe", &err);
  clReleaseProgram (program);
  return err == CL_SUCCESS ? kernel : NULL;
}

/* A kernel runs at most one sub-group per work-item, also when the stack
   size estimate lowers its work-group size limit below the device's, as
   this kernel's private array does. */
static int
check_kernel_limit (cl_context ctx, cl_device_id dev)
{
  cl_kernel kernel
    = build_kernel (ctx, dev,
                    "__kernel void\n"
                    "probe (__global int *out)\n"
                    "{\n"
                    "  volatile int a[65536];\n"
                    "  a[get_local_id (0) % 65536] = 1;\n"
                    "  out[get_global_id (0)] = a[out[0] % 65536];\n"
                    "}\n");
  TEST_ASSERT (kernel != NULL);

  size_t kernel_wg_size = 0, kernel_max_sgs = 0;
  CHECK_CL_ERROR (
    clGetKernelWorkGroupInfo (kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
                              sizeof (kernel_wg_size), &kernel_wg_size, NULL));
  CHECK_CL_ERROR (clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_MAX_NUM_SUB_GROUPS, 0, NULL,
    sizeof (kernel_max_sgs), &kernel_max_sgs, NULL));
  printf ("kernel limit: WG %zu, max SGs %zu\n", kernel_wg_size,
          kernel_max_sgs);
  TEST_ASSERT (kernel_max_sgs <= kernel_wg_size);

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  return CL_SUCCESS;
}

/* With reqd_work_group_size the kernel runs only that shape, so its
   sub-group count is the kernel maximum, and the inverse query can only
   return that shape when it is one-dimensional. */
static int
check_reqd_wg (cl_context ctx,
               cl_device_id dev,
               cl_command_queue queue,
               size_t reqd_sg,
               size_t x,
               size_t y,
               size_t z)
{
  cl_int err;
  char src[1024];
  char attr[256];
  int len = snprintf (attr, sizeof (attr),
                      "__attribute__ ((reqd_work_group_size (%zu, %zu, %zu)))",
                      x, y, z);
  if (reqd_sg > 0)
    snprintf (attr + len, sizeof (attr) - len,
              " __attribute__ ((intel_reqd_sub_group_size (%zu)))", reqd_sg);
  snprintf (src, sizeof (src), LARGE_SRC_TEMPLATE, attr);
  cl_kernel kernel = build_kernel (ctx, dev, src);
  TEST_ASSERT (kernel != NULL);

  cl_mem buf
    = clCreateBuffer (ctx, CL_MEM_READ_WRITE, sizeof (cl_uint), NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  CHECK_CL_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));

  /* the default sub-group spans the whole x dimension */
  size_t expected = reqd_sg > 0 ? (x * y * z + reqd_sg - 1) / reqd_sg : y * z;
  size_t ls[3] = { x, y, z };
  size_t kernel_max_sgs = 0, host_count = 0;
  CHECK_CL_ERROR (clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_MAX_NUM_SUB_GROUPS, 0, NULL,
    sizeof (kernel_max_sgs), &kernel_max_sgs, NULL));
  CHECK_CL_ERROR (clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, sizeof (ls), ls,
    sizeof (host_count), &host_count, NULL));
  TEST_ASSERT (kernel_max_sgs == expected);
  TEST_ASSERT (host_count == expected);

  cl_uint dev_num = 0;
  CHECK_CL_ERROR (
    clEnqueueNDRangeKernel (queue, kernel, 3, NULL, ls, ls, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (
    queue, buf, CL_TRUE, 0, sizeof (cl_uint), &dev_num, 0, NULL, NULL));
  TEST_ASSERT (dev_num == expected);

  size_t wishes[] = { 1, expected, expected + 1 };
  for (size_t w = 0; w < sizeof (wishes) / sizeof (wishes[0]); ++w)
    {
      size_t nd[3] = { 42, 42, 42 };
      CHECK_CL_ERROR (clGetKernelSubGroupInfo (
        kernel, dev, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
        sizeof (wishes[w]), &wishes[w], sizeof (nd), nd, NULL));
      if (y == 1 && z == 1 && wishes[w] == expected)
        TEST_ASSERT (nd[0] == x && nd[1] == 1 && nd[2] == 1);
      else
        TEST_ASSERT (nd[0] == 0 && nd[1] == 0 && nd[2] == 0);
    }

  CHECK_CL_ERROR (clReleaseMemObject (buf));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  return CL_SUCCESS;
}

/* Malformed inputs must be rejected rather than read past. */
static int
check_invalid_inputs (cl_context ctx, cl_device_id dev)
{
  char src[1024];
  snprintf (src, sizeof (src), LARGE_SRC_TEMPLATE, "");
  cl_kernel kernel = build_kernel (ctx, dev, src);
  TEST_ASSERT (kernel != NULL);

  size_t ls[3] = { 8, 1, 1 }, out[3];
  cl_int err = clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, sizeof (size_t) + 1,
    ls, sizeof (size_t), out, NULL);
  TEST_ASSERT (err == CL_INVALID_VALUE);
  cl_uint small_count = 1;
  err = clGetKernelSubGroupInfo (
    kernel, dev, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
    sizeof (small_count), &small_count, sizeof (out), out, NULL);
  TEST_ASSERT (err == CL_INVALID_VALUE);

  CHECK_CL_ERROR (clReleaseKernel (kernel));
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

  char extensions[4096];
  CHECK_CL_ERROR (clGetDeviceInfo (dev, CL_DEVICE_EXTENSIONS,
                                   sizeof (extensions), extensions, NULL));

  /* An application reaches clGetKernelSubGroupInfoKHR by name; without an
     ICD loader, clGetExtensionFunctionAddressForPlatform must return it. */
  if (strstr (extensions, "cl_khr_subgroups") != NULL)
    {
      get_info_khr
        = (clGetKernelSubGroupInfoKHR_t *)
          clGetExtensionFunctionAddressForPlatform (
            platform, "clGetKernelSubGroupInfoKHR");
      TEST_ASSERT (get_info_khr != NULL);
    }

  CHECK_CL_ERROR (check_variant (ctx, dev, queue, 0));
  CHECK_CL_ERROR (check_large_wg (ctx, dev, queue, max_num_sub_groups, 0));
  CHECK_CL_ERROR (check_kernel_limit (ctx, dev));
  CHECK_CL_ERROR (check_invalid_inputs (ctx, dev));
  CHECK_CL_ERROR (check_reqd_wg (ctx, dev, queue, 0, 4, 8, 2));
  CHECK_CL_ERROR (check_reqd_wg (ctx, dev, queue, 0, 64, 1, 1));

  /* only test intel_reqd_sub_group_size where supported */
  if (strstr (extensions, "cl_intel_required_subgroup_size") != NULL)
    {
      CHECK_CL_ERROR (check_variant (ctx, dev, queue, 1));
      CHECK_CL_ERROR (check_large_wg (ctx, dev, queue, max_num_sub_groups, 1));
      CHECK_CL_ERROR (
        check_reqd_wg (ctx, dev, queue, SMALL_REQD_SG_SIZE, 64, 1, 1));
      CHECK_CL_ERROR (
        check_reqd_wg (ctx, dev, queue, SMALL_REQD_SG_SIZE, 20, 2, 1));
    }

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
