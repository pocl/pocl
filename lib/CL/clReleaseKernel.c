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
  unsigned i;

  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);
  POCL_RELEASE_OBJECT (kernel, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release kernel %p  %d\n", kernel, new_refcount);

  if (new_refcount == 0)
    {
      POCL_MSG_PRINT_REFCOUNTS ("Free kernel %p\n", kernel);
      cl_program program = kernel->program;
      /* default kernels are not put into the program->kernels linked list */
      if (program != NULL)
        {
          /* Find the kernel in the program's linked list of kernels */
          POCL_LOCK_OBJ (program);
          for (pk = &program->kernels; *pk != NULL; pk = &(*pk)->next)
            {
              if (*pk == kernel)
                break;
            }

          /* The kernel is not on the kernel's program's linked list
             of kernels -- something is wrong */
          assert (*pk != NULL);

          /* Remove the kernel from the program's linked list of
             kernels */
          *pk = (*pk)->next;
          POCL_RELEASE_OBJECT_UNLOCKED (program, new_refcount);
          POCL_MSG_PRINT_REFCOUNTS ("Removed kernel %p from "
                                    "program %p, which now has refs: %d \n",
                                    kernel, kernel->program, new_refcount);
          POCL_UNLOCK_OBJ (program);
        }

      if (kernel->dyn_arguments)
        for (i = 0; i < (kernel->meta->num_args); i++)
          {
            pocl_aligned_free (kernel->dyn_arguments[i].value);
          }

      kernel->name = NULL;
      kernel->meta = NULL;
      POCL_MEM_FREE (kernel->dyn_arguments);
      POCL_DESTROY_OBJECT (kernel);
      POCL_MEM_FREE (kernel);
    }
  
  return CL_SUCCESS;
}
POsym(clReleaseKernel)
