/* vecbuiltin - simple invocation of a builtins on global buffer of values

   Copyright (c) 2025 Michal Babej / Intel Finland Oy

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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "poclu.h"

#define N 8192

extern int exec_vecbuiltin_kernel (cl_context context,
                                   cl_device_id device,
                                   cl_command_queue cmd_queue,
                                   cl_program program,
                                   int n,
                                   int wg_size,
                                   cl_float *srcA,
                                   cl_float *srcB,
                                   cl_float *dst);

int
main (int argc, char **argv)
{
  cl_float *srcA, *srcB;
  cl_float *dst;
  int i, err;

  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_platform_id platform = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;

  err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");

  const char *basename = "vecbuiltin";
  err = poclu_load_program (platform, context, device, basename, 0, 0, NULL,
                            NULL, &program);
  if (err != CL_SUCCESS)
    goto FINISH;

  int l;
  int wg_size = N / 8;

  srcA = (cl_float *)malloc (N * sizeof (cl_float));
  srcB = (cl_float *)malloc (N * sizeof (cl_float));
  dst = (cl_float *)malloc (N * sizeof (cl_float));

  for (i = 0; i < N; ++i)
    {
      srcA[i] = (cl_float)i / (cl_float)N;
      srcB[i] = (cl_float)(N - i) / (cl_float)N;
      dst[i] = (cl_float)0.0f;
    }

  err = 0;

  if (exec_vecbuiltin_kernel (context, device, queue, program, N, wg_size,
                              srcA, srcB, dst))
    {
      printf ("Error running the tests\n");
      err = 1;
      goto FINISH;
    }

  for (i = 0; i < N; ++i)
    {
      float ref = sinf (srcA[i]) + cosf (srcB[i]) + log2f (125.0f + srcA[i])
                  + powf (srcA[i], 1.5f) + expf (srcA[i]) + exp2f (srcB[i])
                  + fabsf (srcA[i]) + fmaf (srcA[i], 4.0f, srcB[i])
                  + fmaxf (srcA[i], srcB[i]) + fminf (srcA[i], srcB[i])
                  + log10f (125.0f + srcA[i]) + log (125.0f + srcA[i])
                  + rintf (srcA[i]) + round (srcB[i]) + sqrtf (srcB[i])
                  + ceilf (srcB[i]) + tanf (srcA[i]) + powf (srcA[i], 4.0f)
                  + floorf (srcA[i]) + truncf (srcB[i]);
      if (fabsf (ref - dst[i]) > 1.0f)
        {
          printf ("%d FAIL: %f + %f != %f\n", i, srcA[i], srcB[i], dst[i]);
          err = 1;
          goto FINISH;
        }
    }
  free (srcA);
  free (srcB);
  free (dst);

  printf ("OK\n");

FINISH:
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  return err;
}
