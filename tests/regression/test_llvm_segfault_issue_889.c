/* This reduced kernel (originally from clblas invoked by libgpuarray
   tests) triggers a segmentation fault in llvm (10-12) on some platforms.
   https://bugs.debian.org/975931

   Copyright (c) 2021 pocl developers

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

#include "poclu.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char* source =
"__kernel void Sdot_kernel(__global float *_X, __global float *_Y, __global float *scratchBuff,\n"
"                          uint N, uint offx, int incx, uint offy, int incy, int doConj)\n"
"{\n"
"__global float *X = _X + offx;\n"
"__global float *Y = _Y + offy;\n"
"float dotP = (float) 0.0;\n"
"if ( incx < 0 ) {\n"
"X = X + (N - 1) * abs(incx);\n"
"}\n"
"if ( incy < 0 ) {\n"
"Y = Y + (N - 1) * abs(incy);\n"
"}\n"
"int gOffset;\n"
"for( gOffset=(get_global_id(0) * 4); (gOffset + 4 - 1)<N; gOffset+=( get_global_size(0) * 4 ) )\n"
"{\n"
"float4 vReg1, vReg2, res;\n"
"vReg1 = (float4)(  (X + (gOffset*incx))[0 + ( incx * 0)],  (X + (gOffset*incx))[0 + ( incx * 1)],  (X + (gOffset*incx))[0 + ( incx * 2)],  (X + (gOffset*incx))[0 + ( incx * 3)]);\n"
"vReg2 = (float4)(  (Y + (gOffset*incy))[0 + ( incy * 0)],  (Y + (gOffset*incy))[0 + ( incy * 1)],  (Y + (gOffset*incy))[0 + ( incy * 2)],  (Y + (gOffset*incy))[0 + ( incy * 3)]);\n"
"res =  vReg1 *  vReg2 ;\n"
"dotP +=  res .S0 +  res .S1 +  res .S2 +  res .S3;\n"
"}\n"
"for( ; gOffset<N; gOffset++ )\n"
"{\n"
"float sReg1, sReg2, res;\n"
"sReg1 = X[gOffset * incx];\n"
"sReg2 = Y[gOffset * incy];\n"
"res =  sReg1 *  sReg2 ;\n"
"dotP =  dotP +  res ;\n"
"}\n"
"__local float p1753 [ 64 ];\n"
"uint QKiD0 = get_local_id(0);\n"
"p1753 [ QKiD0 ] =  dotP ;\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if( QKiD0 < 32 ) {\n"
"p1753 [ QKiD0 ] = p1753 [ QKiD0 ] + p1753 [ QKiD0 + 32 ];\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if( QKiD0 < 16 ) {\n"
"p1753 [ QKiD0 ] = p1753 [ QKiD0 ] + p1753 [ QKiD0 + 16 ];\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if( QKiD0 < 8 ) {\n"
"p1753 [ QKiD0 ] = p1753 [ QKiD0 ] + p1753 [ QKiD0 + 8 ];\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if( QKiD0 < 4 ) {\n"
"p1753 [ QKiD0 ] = p1753 [ QKiD0 ] + p1753 [ QKiD0 + 4 ];\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if( QKiD0 < 2 ) {\n"
"p1753 [ QKiD0 ] = p1753 [ QKiD0 ] + p1753 [ QKiD0 + 2 ];\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if( QKiD0 == 0 ) {\n"
"dotP  = p1753 [0] + p1753 [1];\n"
"}\n"
"if( (get_local_id(0)) == 0 ) {\n"
"scratchBuff[ get_group_id(0) ] = dotP;\n"
"}\n"
"}\n"
;

int
main ()
{
  cl_int err;
  cl_context context;
  cl_device_id device;
  cl_command_queue command_queue;
  poclu_get_any_device (&context, &device, &command_queue);

  cl_program program
      = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");

  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, "-g", NULL, NULL));

  size_t binsizes[32];
  size_t nbinaries;
  CHECK_CL_ERROR (clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES,
                                    sizeof (binsizes), binsizes, &nbinaries));
  for (size_t i = 0; i < nbinaries; ++i)
    printf ("binary size [%zd]: %zd\n", i, binsizes[i]);

  CHECK_CL_ERROR (clReleaseProgram (program));

  CHECK_CL_ERROR (clReleaseContext (context));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
