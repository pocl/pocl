/* example1 - Simple example from OpenCL specification.

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
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

#define N 128

#ifdef __cplusplus
#  define CALLAPI "C"
#else 
#  define CALLAPI
#endif

extern CALLAPI int exec_dot_product_kernel (const char *program_source, 
            int n, void *srcA, void *srcB, void *dst);

int
main (void)
{
  FILE *source_file;
  char *source;
  int source_size;
  cl_float4 *srcA, *srcB;
  cl_float *dst;
  int i;

  source_file = fopen("example1.cl", "r");
  if (source_file == NULL) 
    source_file = fopen (SRCDIR "/example1.cl", "r");

  assert(source_file != NULL && "example1.cl not found!");

  fseek (source_file, 0, SEEK_END);
  source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  source = (char *) malloc (source_size +1 );
  assert (source != NULL);

  fread (source, source_size, 1, source_file);
  source[source_size] = '\0';

  fclose (source_file);

  srcA = (cl_float4 *) malloc (N * sizeof (cl_float4));
  srcB = (cl_float4 *) malloc (N * sizeof (cl_float4));
  dst = (cl_float *) malloc (N * sizeof (cl_float));

  for (i = 0; i < N; ++i)
    {
      srcA[i].s[0] = (cl_float)i;
      srcA[i].s[1] = (cl_float)i;
      srcA[i].s[2] = (cl_float)i;
      srcA[i].s[3] = (cl_float)i;
      srcB[i].s[0] = (cl_float)i;
      srcB[i].s[1] = (cl_float)i;
      srcB[i].s[2] = (cl_float)i;
      srcB[i].s[3] = (cl_float)i;
    }

  if (exec_dot_product_kernel (source, N, srcA, srcB, dst))
    {
      printf ("Error running the tests\n");
      return -1;
    }

  for (i = 0; i < 4; ++i)
    {
      printf ("(%f, %f, %f, %f) . (%f, %f, %f, %f) = %f\n",
        srcA[i].s[0], srcA[i].s[1], srcA[i].s[2], srcA[i].s[3],
        srcB[i].s[0], srcB[i].s[1], srcB[i].s[2], srcB[i].s[3],
        dst[i]);
      if (srcA[i].s[0] * srcB[i].s[0] +
          srcA[i].s[1] * srcB[i].s[1] +
          srcA[i].s[2] * srcB[i].s[2] +
          srcA[i].s[3] * srcB[i].s[3] != dst[i])
        {
          printf ("FAIL\n");
          return -1;
        }
    }

  free (source);
  free (srcA);
  free (srcB);
  free (dst);


  printf ("OK\n");
  return 0;
}
