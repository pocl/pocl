/* boxadd - 3D box value addition using work-items.

   Copyright (c) 2018-2019 Pekka Jääskeläinen

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
#include <poclu.h>

#define SIZEX (1230)
#define SIZEY (412)
#define SIZEZ (8)
#define DATA_SIZE (SIZEX*SIZEY*SIZEZ)

#ifdef __cplusplus
#  define CALLAPI "C"
#else
#  define CALLAPI
#endif

extern CALLAPI int
exec_matadd_kernel (cl_context context, cl_device_id device,
		    cl_command_queue cmd_queue, cl_program program,
		    int sizeX, int sizeY, int sizeZ,
		    cl_float *srcA, cl_float *srcB, cl_float *dst);

int
main (int argc, char **argv)
{
  cl_float *srcA, *srcB;
  cl_float *dst;
  int i, j, k, err;

  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_platform_id platform = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;

  err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");

  const char *basename = "boxadd";
  err = poclu_load_program (context, device, basename, 0, 0, 0,
			    NULL, NULL, &program);
  if (err != CL_SUCCESS)
    goto FINISH;

  srcA = (cl_float *) malloc (DATA_SIZE * sizeof (cl_float));
  srcB = (cl_float *) malloc (DATA_SIZE * sizeof (cl_float));
  dst = (cl_float *) malloc (DATA_SIZE * sizeof (cl_float));

  for (k = 0; k < SIZEZ; ++k)
    for (j = 0; j < SIZEY; ++j)
      for (i = 0; i < SIZEX; ++i)
	{
	  int indx = k*SIZEX*SIZEY + j*SIZEX + i;
	  srcA[indx] = (cl_float)indx;
	  srcB[indx] = (cl_float)(DATA_SIZE - indx);
	  dst[indx] = (cl_float)-1;
	}

  err = 0;

  if (exec_matadd_kernel (context, device, queue, program,
			  SIZEX, SIZEY, SIZEZ,
			  srcA, srcB, dst))
    {
      printf ("Error running the tests\n");
      err = 1;
      goto FINISH;
    }

  for (k = 0; k < SIZEZ; ++k)
    for (j = 0; j < SIZEY; ++j)
      for (i = 0; i < SIZEX; ++i)
	{
	  int indx = k*SIZEX*SIZEY + j*SIZEX + i;
	  if ((int)srcA[indx] + (int)srcB[indx] != (int)dst[indx])
	    {
	      printf ("%d FAIL: %f + %f != %f\n", indx,
		      srcA[indx], srcB[indx], dst[indx]);
	      err = 1;
	      goto FINISH;
	    }
	}

  printf ("OK\n");

FINISH:
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));
  CHECK_CL_ERROR (clReleaseContext (context));

  return err;
}
