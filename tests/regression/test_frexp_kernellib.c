// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "pocl_opencl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* frexp, and the double powr/rootn built on it, must be correct once the
   work-item loop vectorizes. llvm.frexp on vectors is miscompiled on x86 below
   AVX2 (llvm/llvm-project#224127): the exponent comes back as the mantissa's
   high bits, which made powr(2, 3) and rootn(8, 3) return inf. An optional
   argument selects the kernel library variant (POCL_KERNELLIB_NAME). */

#define WG_SIZE 64
#define N 256

#define CHECK_ERROR(err)                                                      \
  do                                                                          \
    {                                                                         \
      if ((err) != CL_SUCCESS)                                                \
        {                                                                     \
          fprintf (stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__,     \
                   __LINE__);                                                 \
          goto error;                                                         \
        }                                                                     \
    }                                                                         \
  while (0)

static const char KernelSource[]
  = "__kernel void test_fp32(__global const float *x, __global float *mant,\n"
    "                        __global int *expo) {\n"
    "  size_t i = get_global_id(0);\n"
    "  int e;\n"
    "  mant[i] = frexp(x[i], &e);\n"
    "  expo[i] = e;\n"
    "}\n"
    "#ifdef cl_khr_fp16\n"
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void test_fp16(__global const float *x, __global float *mant,\n"
    "                        __global int *expo) {\n"
    "  size_t i = get_global_id(0);\n"
    "  int e;\n"
    "  mant[i] = (float)frexp((half)x[i], &e);\n"
    "  expo[i] = e;\n"
    "}\n"
    "#endif\n"
    "#ifdef cl_khr_fp64\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void test_fp64(__global const double *x,\n"
    "                        __global const double *y,\n"
    "                        __global const int *n,\n"
    "                        __global double *mant, __global int *expo,\n"
    "                        __global double *pw, __global double *rt) {\n"
    "  size_t i = get_global_id(0);\n"
    "  int e;\n"
    "  mant[i] = frexp(x[i], &e);\n"
    "  expo[i] = e;\n"
    "  pw[i] = powr(x[i], y[i]);\n"
    "  rt[i] = rootn(x[i], n[i]);\n"
    "}\n"
    /* converting the exponent back to double used to fail instruction
       selection instead of miscompiling, so it gets its own kernel */
    "__kernel void test_fp64_sum(__global const double *x,\n"
    "                            __global double *sum) {\n"
    "  size_t i = get_global_id(0);\n"
    "  int e;\n"
    "  double m = frexp(x[i], &e);\n"
    "  sum[i] = m + (double)e;\n"
    "}\n"
    "#endif\n";

static int
close_enough (double got, double want)
{
  if (isnan (want))
    return isnan (got);
  if (isinf (want) || want == 0.0)
    return got == want;
  return fabs (got - want) <= 1e-12 * fabs (want);
}

static double
input (int i)
{
  /* positive normals across a wide exponent range, plus a few subnormals */
  if (i % 16 == 15)
    return ldexp (1.0 + i / (double)N, -1060 + i % 40);
  return ldexp (1.0 + i / (double)N, (i * 37) % 120 - 60);
}

int
main (int argc, char **argv)
{
  cl_int err;
  cl_uint num_platforms;
  cl_platform_id *platforms = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel k16 = NULL, k32 = NULL, k64 = NULL, k64sum = NULL;
  cl_mem bufs[8] = { NULL };
  const size_t global = N, local = WG_SIZE;
  int result = 1, errors = 0;

  if (argc > 1)
    {
#ifdef _WIN32
      _putenv_s ("POCL_KERNELLIB_NAME", argv[1]);
#else
      setenv ("POCL_KERNELLIB_NAME", argv[1], 1);
#endif
    }

  err = clGetPlatformIDs (0, NULL, &num_platforms);
  CHECK_ERROR (err);
  if (num_platforms == 0)
    {
      puts ("SKIP: no OpenCL platforms");
      return 77;
    }
  platforms = malloc (num_platforms * sizeof (*platforms));
  if (platforms == NULL)
    goto error;
  err = clGetPlatformIDs (num_platforms, platforms, NULL);
  CHECK_ERROR (err);

  cl_device_id device;
  err = clGetDeviceIDs (platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  CHECK_ERROR (err);
  context = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR (err);
  const cl_queue_properties properties[] = { 0 };
  queue
    = clCreateCommandQueueWithProperties (context, device, properties, &err);
  CHECK_ERROR (err);

  cl_device_fp_config fp64 = 0, fp16 = 0;
  CHECK_ERROR (clGetDeviceInfo (device, CL_DEVICE_DOUBLE_FP_CONFIG,
                                sizeof (fp64), &fp64, NULL));
  /* without cl_khr_fp16, this query fails with CL_INVALID_VALUE */
  if (clGetDeviceInfo (device, CL_DEVICE_HALF_FP_CONFIG, sizeof (fp16), &fp16,
                       NULL)
      != CL_SUCCESS)
    fp16 = 0;

  const char *sources[] = { KernelSource };
  program = clCreateProgramWithSource (context, 1, sources, NULL, &err);
  CHECK_ERROR (err);
  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t log_size = 0;
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                             &log_size);
      char *log = malloc (log_size);
      if (log != NULL)
        {
          clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                                 log_size, log, NULL);
          fprintf (stderr, "Build error:\n%s\n", log);
          free (log);
        }
      goto error;
    }

  /* float frexp */
  {
    static cl_float x[N], mant[N];
    static cl_int expo[N];
    for (int i = 0; i < N; ++i)
      x[i] = (float)ldexp (1.0 + i / (double)N, (i * 37) % 200 - 100);

    k32 = clCreateKernel (program, "test_fp32", &err);
    CHECK_ERROR (err);
    bufs[0] = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof (x), x, &err);
    CHECK_ERROR (err);
    bufs[1]
      = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (mant), NULL, &err);
    CHECK_ERROR (err);
    bufs[2]
      = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (expo), NULL, &err);
    CHECK_ERROR (err);
    for (cl_uint a = 0; a < 3; ++a)
      CHECK_ERROR (clSetKernelArg (k32, a, sizeof (cl_mem), &bufs[a]));
    CHECK_ERROR (clEnqueueNDRangeKernel (queue, k32, 1, NULL, &global, &local,
                                         0, NULL, NULL));
    CHECK_ERROR (clEnqueueReadBuffer (queue, bufs[1], CL_TRUE, 0,
                                      sizeof (mant), mant, 0, NULL, NULL));
    CHECK_ERROR (clEnqueueReadBuffer (queue, bufs[2], CL_TRUE, 0,
                                      sizeof (expo), expo, 0, NULL, NULL));
    for (int i = 0; i < N; ++i)
      {
        int e;
        float m = frexpf (x[i], &e);
        if (mant[i] != m || expo[i] != e)
          {
            if (errors++ < 10)
              fprintf (stderr,
                       "float frexp(%a) = (%a, %d), expected (%a, %d)\n", x[i],
                       mant[i], expo[i], m, e);
          }
      }
    for (int a = 0; a < 3; ++a)
      {
        clReleaseMemObject (bufs[a]);
        bufs[a] = NULL;
      }
  }

  /* half frexp, including half subnormals */
  if (fp16 == 0)
    puts ("no cl_khr_fp16, skipping the half checks");
  else
    {
      static cl_float x[N], mant[N];
      static cl_int expo[N];
      /* values that are exact in half: subnormals k * 2^-24, and normals
         with a 6-bit mantissa */
      for (int i = 0; i < N; ++i)
        x[i] = i < 64 ? (float)ldexp (i + 1, -24)
                      : (float)ldexp (1.0 + (i % 64) / 64.0, i % 30 - 14);

      k16 = clCreateKernel (program, "test_fp16", &err);
      CHECK_ERROR (err);
      bufs[0] = clCreateBuffer (
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (x), x, &err);
      CHECK_ERROR (err);
      bufs[1] = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (mant),
                                NULL, &err);
      CHECK_ERROR (err);
      bufs[2] = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (expo),
                                NULL, &err);
      CHECK_ERROR (err);
      for (cl_uint a = 0; a < 3; ++a)
        CHECK_ERROR (clSetKernelArg (k16, a, sizeof (cl_mem), &bufs[a]));
      CHECK_ERROR (clEnqueueNDRangeKernel (queue, k16, 1, NULL, &global,
                                           &local, 0, NULL, NULL));
      CHECK_ERROR (clEnqueueReadBuffer (queue, bufs[1], CL_TRUE, 0,
                                        sizeof (mant), mant, 0, NULL, NULL));
      CHECK_ERROR (clEnqueueReadBuffer (queue, bufs[2], CL_TRUE, 0,
                                        sizeof (expo), expo, 0, NULL, NULL));
      for (int i = 0; i < N; ++i)
        {
          int e;
          /* the inputs are exact in half, whose subnormals are float
             normals, so float frexp is the reference */
          float m = frexpf (x[i], &e);
          if (mant[i] != m || expo[i] != e)
            {
              if (errors++ < 10)
                fprintf (stderr,
                         "half frexp(%a) = (%a, %d), expected (%a, %d)\n",
                         x[i], mant[i], expo[i], m, e);
            }
        }
      for (int a = 0; a < 3; ++a)
        {
          clReleaseMemObject (bufs[a]);
          bufs[a] = NULL;
        }
    }

  /* double frexp, powr, rootn */
  if (fp64 == 0)
    puts ("no cl_khr_fp64, skipping the double checks");
  else
    {
      static cl_double x[N], y[N], mant[N], sum[N], pw[N], rt[N];
      static cl_int n[N], expo[N];
      for (int i = 0; i < N; ++i)
        {
          x[i] = input (i);
          y[i] = (double)(i % 7) - 3.0;
          n[i] = i % 4 + 1;
        }
      x[0] = 2.0, y[0] = 3.0, n[0] = 3; /* powr(2, 3) = 8, rootn(2, 3) */
      x[1] = 8.0, y[1] = 2.0, n[1] = 3; /* powr(8, 2) = 64, rootn(8, 3) = 2 */

      k64 = clCreateKernel (program, "test_fp64", &err);
      CHECK_ERROR (err);
      bufs[0] = clCreateBuffer (
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (x), x, &err);
      CHECK_ERROR (err);
      bufs[1] = clCreateBuffer (
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (y), y, &err);
      CHECK_ERROR (err);
      bufs[2] = clCreateBuffer (
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (n), n, &err);
      CHECK_ERROR (err);
      const size_t out_sizes[5] = { sizeof (mant), sizeof (expo), sizeof (pw),
                                    sizeof (rt), sizeof (sum) };
      for (int a = 0; a < 5; ++a)
        {
          bufs[3 + a] = clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                        out_sizes[a], NULL, &err);
          CHECK_ERROR (err);
        }
      for (cl_uint a = 0; a < 7; ++a)
        CHECK_ERROR (clSetKernelArg (k64, a, sizeof (cl_mem), &bufs[a]));
      CHECK_ERROR (clEnqueueNDRangeKernel (queue, k64, 1, NULL, &global,
                                           &local, 0, NULL, NULL));
      void *outs[4] = { mant, expo, pw, rt };
      for (int a = 0; a < 4; ++a)
        CHECK_ERROR (clEnqueueReadBuffer (queue, bufs[3 + a], CL_TRUE, 0,
                                          out_sizes[a], outs[a], 0, NULL,
                                          NULL));

      for (int i = 0; i < N; ++i)
        {
          int e;
          double m = frexp (x[i], &e);
          double want_pw = pow (x[i], y[i]);
          double want_rt = pow (x[i], 1.0 / n[i]);
          if (mant[i] != m || expo[i] != e)
            {
              if (errors++ < 10)
                fprintf (stderr,
                         "double frexp(%a) = (%a, %d), expected (%a, %d)\n",
                         x[i], mant[i], expo[i], m, e);
            }
          if (!close_enough (pw[i], want_pw))
            {
              if (errors++ < 10)
                fprintf (stderr, "powr(%a, %a) = %a, expected %a\n", x[i],
                         y[i], pw[i], want_pw);
            }
          if (!close_enough (rt[i], want_rt))
            {
              if (errors++ < 10)
                fprintf (stderr, "rootn(%a, %d) = %a, expected %a\n", x[i],
                         n[i], rt[i], want_rt);
            }
        }

      k64sum = clCreateKernel (program, "test_fp64_sum", &err);
      CHECK_ERROR (err);
      CHECK_ERROR (clSetKernelArg (k64sum, 0, sizeof (cl_mem), &bufs[0]));
      CHECK_ERROR (clSetKernelArg (k64sum, 1, sizeof (cl_mem), &bufs[7]));
      CHECK_ERROR (clEnqueueNDRangeKernel (queue, k64sum, 1, NULL, &global,
                                           &local, 0, NULL, NULL));
      CHECK_ERROR (clEnqueueReadBuffer (queue, bufs[7], CL_TRUE, 0,
                                        sizeof (sum), sum, 0, NULL, NULL));
      for (int i = 0; i < N; ++i)
        {
          int e;
          double m = frexp (x[i], &e);
          if (sum[i] != m + (double)e)
            {
              if (errors++ < 10)
                fprintf (stderr,
                         "frexp(%a) mantissa + exponent = %a, "
                         "expected %a\n",
                         x[i], sum[i], m + (double)e);
            }
        }
    }

  if (errors)
    {
      fprintf (stderr, "%d mismatches\n", errors);
      goto error;
    }
  puts ("OK");
  result = 0;

error:
  free (platforms);
  for (int a = 0; a < 8; ++a)
    if (bufs[a] != NULL)
      clReleaseMemObject (bufs[a]);
  if (k64sum != NULL)
    clReleaseKernel (k64sum);
  if (k64 != NULL)
    clReleaseKernel (k64);
  if (k32 != NULL)
    clReleaseKernel (k32);
  if (k16 != NULL)
    clReleaseKernel (k16);
  if (program != NULL)
    clReleaseProgram (program);
  if (queue != NULL)
    clReleaseCommandQueue (queue);
  if (context != NULL)
    clReleaseContext (context);
  return result;
}
