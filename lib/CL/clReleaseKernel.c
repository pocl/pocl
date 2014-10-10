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
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseKernel)(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  cl_kernel *pk;
  int i;

  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);

  POCL_RELEASE_OBJECT (kernel, new_refcount);

  if (new_refcount == 0)
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
          POname(clReleaseProgram) (kernel->program);
        }
      
      POCL_MEM_FREE(kernel->function_name);
      POCL_MEM_FREE(kernel->name);

      for (i = 0; i < kernel->num_args; i++)
        {
          struct pocl_argument *p = &(kernel->dyn_arguments[i]);
          if (p->value != NULL)
            {
              pocl_aligned_free (p->value);
              p->value = NULL;
            }
        }

      POCL_MEM_FREE(kernel->dyn_arguments);
      POCL_MEM_FREE(kernel->reqd_wg_size);
      POCL_MEM_FREE(kernel);
    }
  
  return CL_SUCCESS;
}
POsym(clReleaseKernel)
