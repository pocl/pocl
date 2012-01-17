/* OpenCL runtime library: clReleaseKernel()

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
clReleaseKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
  cl_kernel *pk;

  POCL_RELEASE_OBJECT (kernel);

  if (kernel->pocl_refcount == 0)
    {

      if (kernel->program != NULL)
        {
          /* Find the kernel in the program's linked list of kernels */
          for (pk=&kernel->program->kernels; *pk != NULL; pk = &(*pk)->next)
            {
              if (*pk == kernel) break;
            }
          if (*pk == NULL)
            {
              /* The kernel is not on the kernel's program's linked list
                 of kernels -- something is wrong */
              return CL_INVALID_VALUE;
            }
          
          /* Remove the kernel from the program's linked list of
             kernels */
          *pk = (*pk)->next;
        }
      
      free ((char*)kernel->function_name);
      free (kernel);
    }
  
  return CL_SUCCESS;
}
