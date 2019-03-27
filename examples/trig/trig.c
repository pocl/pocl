/* trig - Trigonometric functions.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pocl_opencl.h"

#define N 8

extern int exec_trig_kernel (const char *program_source, 
                             int n, void *srcA, void *dst);

int
main (void)
{
  FILE *source_file;
  char *source;
  int source_size;
  cl_float4 *srcA;
  cl_float4 *dst;
  cl_float *dstS;
  int i;

  source_file = fopen("trig.cl", "r");
  if (source_file == NULL) 
    source_file = fopen (SRCDIR "/trig.cl", "r");

  assert(source_file != NULL && "trig.cl not found!");

  fseek (source_file, 0, SEEK_END);
  source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  source = (char *) malloc (source_size + 1);
  assert (source != NULL);

  fread (source, source_size, 1, source_file);
  source[source_size] = '\0';

  fclose (source_file);

  srcA = (cl_float4 *) malloc (N * sizeof (cl_float4));
  dst = (cl_float4 *) malloc (N * sizeof (cl_float4));
  dstS = (cl_float *) malloc (N * sizeof (cl_float));

  for (i = 0; i < N; ++i)
    {
      srcA[i].s[0] = (cl_float)i;
      srcA[i].s[1] = (cl_float)i;
      srcA[i].s[2] = (cl_float)i;
      srcA[i].s[3] = (cl_float)i;
      switch (i % 5) {
      case 0: dstS[i] = cosf((float)i); break;
      case 1: dstS[i] = fabsf((float)i) + 7.3f; break;
      case 2: dstS[i] = sinf((float)i); break;
      case 3: dstS[i] = sqrtf((float)i); break;
      case 4: dstS[i] = tanf((float)i); break;
      }
    }

  if (exec_trig_kernel (source, N, srcA, dst) < 0)
    {
      printf("Failed to run the kernel.\n");
      return -1;
    }

  for (i = 0; i < N; ++i)
    {
      if (fabsf(dst[i].s[0] - dstS[i]) > 1.0e-6f ||
          fabsf(dst[i].s[1] - dstS[i]) > 1.0e-6f ||
          fabsf(dst[i].s[2] - dstS[i]) > 1.0e-6f ||
          fabsf(dst[i].s[3] - dstS[i]) > 1.0e-6f)
  {
          printf ("input:    [%.7f, %.7f, %.7f, %.7f]\n"
                  "output:   [%.7f, %.7f, %.7f, %.7f]\n"
                  "expected: [%.7f, %.7f, %.7f, %.7f]\n",
                  srcA[i].s[0], srcA[i].s[1], srcA[i].s[2], srcA[i].s[3],
                  dst[i].s[0], dst[i].s[1], dst[i].s[2], dst[i].s[3],
                  dstS[i], dstS[i], dstS[i], dstS[i]);
    printf ("FAIL\n");
    return -1;
  }
    }

  printf ("OK\n");

  free (srcA);
  free (dst);
  free (dstS);

  return 0;
}
