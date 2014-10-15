/* OpenCL runtime library: clReleaseContext()

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

#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseContext)(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  if (!context->valid)
    return CL_INVALID_CONTEXT;

  POCL_RELEASE_OBJECT(context, new_refcount);
  if (new_refcount == 0)
    {
      /* The context holds references to all its devices,
         memory objects, command-queues etc. Release the
         references and let the objects to get freed. */
      /* TODO: call the corresponding clRelease* functions
         for all the referred objects. */
      int i;
      for (i = 0; i < context->num_devices; ++i) 
        {
          POname(clReleaseDevice) (context->devices[i]);
        }   
      POCL_MEM_FREE(context->devices);
      POCL_MEM_FREE(context->properties);
      POCL_MEM_FREE(context);
    }
  return CL_SUCCESS;
}
POsym(clReleaseContext)
