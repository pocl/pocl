/* OpenCL runtime library: clGetPlatformIDs()

   Copyright (c) 2011 Kalle Raiskila 
   
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
#include <string.h>
#include "pocl_cl.h"

#pragma GCC visibility push(hidden)
#ifdef BUILD_ICD
struct _cl_icd_dispatch pocl_dispatch = POCL_ICD_DISPATCH;
struct _cl_platform_id _platforms[1]  = {{&pocl_dispatch}};
#else
struct _cl_platform_id _platforms[1]  = {};
#endif
#pragma GCC visibility pop

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
CL_API_ENTRY cl_int CL_API_CALL
POname(clGetPlatformIDs)(cl_uint           num_entries,
                 cl_platform_id *  platforms,
                 cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{	
  int const num = 1;
  int i;
  
  if (platforms != NULL) {
    if (num_entries < num)
      return CL_INVALID_VALUE;
    
    for (i=0; i<num; ++i)
      platforms[i] = &_platforms[i];
  }
  
  if (num_platforms != NULL)
    *num_platforms = num;
  
  return CL_SUCCESS;
}
POsym(clGetPlatformIDs)
