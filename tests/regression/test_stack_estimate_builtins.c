/* Regression test for the stack-size based CL_KERNEL_WORK_GROUP_SIZE limit
   (HOST_CPU_ENABLE_STACK_SIZE_CHECK) counting the local variables of builtins.

   The kernel library is built at -O0, so builtins like hypot carry well over
   a kilobyte of allocas into the linked program. The kernel compiler optimizes
   them away, but the stack estimate used to count them, limiting kernels that
   need almost no stack to a few hundred work-items on macOS.

   Copyright (c) 2026 Tim Besard

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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

/* builtins: calls builtins with large -O0 frames, but has no private array.
   private_array: a dynamically indexed array that stays in memory, which the
   limit has to keep accounting for.
   private_array_builtins: both, which should need as much stack as
   private_array alone. */
static const char *kernel_source
  = "#define PRIVATE_ARRAY                       \\\n"
    "  size_t gid = get_global_id (0);           \\\n"
    "  long buf[4096];                           \\\n"
    "  for (int i = 0; i < 4096; i++)            \\\n"
    "    buf[i] = (long)gid + i;                 \\\n"
    "  long acc = 0;                             \\\n"
    "  for (int i = 0; i < 4096; i++)            \\\n"
    "    acc += buf[buf[i] % 4096];\n"
    "#define BUILTINS(i) \\\n"
    "  (hypot (a[i], b[i]) + tgamma (b[i]) + atan2 (a[i], b[i]))\n"
    "kernel void builtins (global float *a, global float *b)\n"
    "{\n"
    "  size_t i = get_global_id (0);\n"
    "  a[i] = BUILTINS (i);\n"
    "}\n"
    "kernel void private_array (global long *out)\n"
    "{\n"
    "  PRIVATE_ARRAY\n"
    "  out[gid] = acc;\n"
    "}\n"
    "kernel void private_array_builtins (global long *out,\n"
    "                                    global float *a, global float *b)\n"
    "{\n"
    "  PRIVATE_ARRAY\n"
    "  out[gid] = acc + (long)BUILTINS (gid);\n"
    "}\n";

static int
get_wg_size (cl_program program,
             cl_device_id device,
             const char *name,
             size_t *wg_size)
{
  cl_int err;
  cl_kernel kernel = clCreateKernel (program, name, &err);
  CHECK_ERROR (err);
  err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof (*wg_size), wg_size, NULL);
  CHECK_ERROR (err);
  CHECK_ERROR (clReleaseKernel (kernel));
  return EXIT_SUCCESS;
}

int
main (void)
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;

  cl_int err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_ERROR (err);

  cl_device_type type;
  err = clGetDeviceInfo (device, CL_DEVICE_TYPE, sizeof (type), &type, NULL);
  CHECK_ERROR (err);
  if (!(type & CL_DEVICE_TYPE_CPU))
    {
      printf ("SKIP: the test targets CPU devices\n");
      return 77;
    }

  size_t max_wg_size;
  err = clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof (max_wg_size), &max_wg_size, NULL);
  CHECK_ERROR (err);

  cl_program program
    = clCreateProgramWithSource (context, 1, &kernel_source, NULL, &err);
  CHECK_ERROR (err);
  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  CHECK_ERROR (err);

  size_t builtins_wg_size, private_wg_size, private_builtins_wg_size;
  if (get_wg_size (program, device, "builtins", &builtins_wg_size)
      || get_wg_size (program, device, "private_array", &private_wg_size)
      || get_wg_size (program, device, "private_array_builtins",
                      &private_builtins_wg_size))
    return EXIT_FAILURE;
  printf ("CL_KERNEL_WORK_GROUP_SIZE: builtins %zu, private_array %zu, "
          "private_array_builtins %zu, device max %zu\n",
          builtins_wg_size, private_wg_size, private_builtins_wg_size,
          max_wg_size);

  CHECK_ERROR (clReleaseProgram (program));
  CHECK_ERROR (clReleaseCommandQueue (queue));
  CHECK_ERROR (clReleaseContext (context));
  CHECK_ERROR (clUnloadPlatformCompiler (platform));

  /* 32 KiB of private memory per work-item fits no stack budget at the
     device maximum, so an unlimited private_array means the device doesn't
     limit work-group sizes by stack use. */
  if (private_wg_size == max_wg_size)
    {
      printf ("SKIP: no stack-size based work-group size limit\n");
      return 77;
    }

  if (builtins_wg_size != max_wg_size)
    {
      printf ("FAIL: builtins limited to %zu work-items\n", builtins_wg_size);
      return EXIT_FAILURE;
    }
  /* Mostly independent of the stack size: stack charged to the builtins
     lowers the limit below that of the array alone, unless it's small enough
     to vanish in the rounding of the division. */
  if (private_builtins_wg_size != private_wg_size)
    {
      printf ("FAIL: builtins lower the limit from %zu to %zu work-items\n",
              private_wg_size, private_builtins_wg_size);
      return EXIT_FAILURE;
    }
  printf ("OK\n");
  return EXIT_SUCCESS;
}
