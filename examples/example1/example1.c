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

#define N 4

extern int exec_dot_product_kernel (const char *program_source, 
				    int n, void *srcA, void *srcB, void *dst);

int
main (void)
{
  FILE *source_file;
  char *source;
  int source_size;
  cl_float4 *srcA, *srcB;
  cl_float *dst;
  int ierr;
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
      srcA[i].x = i;
      srcA[i].y = i;
      srcA[i].z = i;
      srcA[i].w = i;
      srcB[i].x = i;
      srcB[i].y = i;
      srcB[i].z = i;
      srcB[i].w = i;
    }

  ierr = exec_dot_product_kernel (source, N, srcA, srcB, dst);
  if (ierr) printf ("ERROR\n");

  for (i = 0; i < N; ++i)
    {
      printf ("(%f, %f, %f, %f) . (%f, %f, %f, %f) = %f\n",
	      srcA[i].x, srcA[i].y, srcA[i].z, srcA[i].w,
	      srcB[i].x, srcB[i].y, srcB[i].z, srcB[i].w,
	      dst[i]);
      if (srcA[i].x * srcB[i].x +
	  srcA[i].y * srcB[i].y +
	  srcA[i].z * srcB[i].z +
	  srcA[i].w * srcB[i].w != dst[i])
	{
	  printf ("FAIL\n");
	  return -1;
	}
    }

  printf ("OK\n");
  return 0;
}
