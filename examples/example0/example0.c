/* example0 - Simple example from OpenCL 1.0 specification, modified

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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "poclu.h"

#define N 128

#ifdef __cplusplus
#define CALLAPI "C"
#else
#define CALLAPI
#endif

extern CALLAPI int exec_integer_mad_kernel (cl_context context,
                                            cl_device_id device,
                                            cl_command_queue cmd_queue,
                                            cl_program program, unsigned n,
                                            cl_uint *srcA, cl_uint *srcB,
                                            cl_uint *dst);

int
main (int argc, char **argv)
{
  cl_uint *srcA, *srcB;
  cl_uint *dst;
  int i, err, spirv, poclbin;

  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_platform_id platform = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;

  srand (time (NULL));

  err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");

  spirv = (argc > 1 && argv[1][0] == 'v');
  poclbin = (argc > 1 && argv[1][0] == 'b');
  const char *explicit_binary_path = (argc > 2) ? argv[2] : NULL;

  const char *basename = "example0";
  err = poclu_load_program (context, device, basename, spirv, poclbin,
                            explicit_binary_path, NULL, &program);
  if (err != CL_SUCCESS)
    goto FINISH;

  srcA = (cl_uint *)malloc (N * sizeof (cl_uint));
  srcB = (cl_uint *)malloc (N * sizeof (cl_uint));
  dst = (cl_uint *)malloc (N * sizeof (cl_uint));

  for (i = 0; i < N; ++i)
    {
      srcA[i] = (cl_uint) (rand () & (0xFF));
      srcB[i] = (cl_uint) (rand () & (0xFF));
      dst[i] = (cl_uint)0;
    }

  err = 0;

  if (exec_integer_mad_kernel (context, device, queue, program, N, srcA, srcB,
                               dst))
    {
      printf ("Error running the tests\n");
      err = 1;
      goto FINISH;
    }

  for (i = 0; i < N; ++i)
    {
      int not_equal = ((srcA[i] * 7 + srcB[i]) != dst[i]);
      printf ("(%u * 7 + %u)  = %u (%s)\n", srcA[i], srcB[i], dst[i],
              (not_equal ? "ERROR" : "OK"));
      if (not_equal)
        err = 1;
    }

  if (err)
    printf ("FAIL\n");
  else
    {
      printf ("PASS\n");

      int generate_output = (argc > 1 && argv[1][0] == 'o');
      if (generate_output)
        {
          FILE *f = fopen ("golden_output.txt", "w");
          for (i = 0; i < N; ++i)
            {
              fprintf (f, "%d\n", dst[i]);
            }
          fclose (f);
        }
    }

FINISH:
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  free(srcA);
  free(srcB);
  free(dst);


  return err;
}
