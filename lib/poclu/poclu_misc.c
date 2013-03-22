/* poclu_misc - misc generic OpenCL helper functions

   Copyright (c) 2013 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "poclu.h"
#include <CL/opencl.h>
#include "config.h"

cl_context
poclu_create_any_context() 
{
  cl_uint i;
  cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id));

  clGetPlatformIDs(1, platforms, &i);
  if (i == 0)
    return (cl_context)0;

  cl_context_properties properties[] = 
    {CL_CONTEXT_PLATFORM, 
     (cl_context_properties)platforms[0], 
     0};

  // create the OpenCL context on any available OCL device 
  cl_context context = clCreateContextFromType(
      properties, 
      CL_DEVICE_TYPE_ALL,
      NULL, NULL, NULL); 

  free (platforms);
  return context;
}
