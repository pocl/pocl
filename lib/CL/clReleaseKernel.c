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

extern unsigned long kernel_c;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseKernel)(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  unsigned i;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  POCL_LOCK_OBJ (kernel);
  POCL_RELEASE_OBJECT_UNLOCKED (kernel, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Kernel %s (%p), Refcount: %d\n",
                            kernel->name, kernel, new_refcount);

  if (new_refcount == 0)
    {
      POCL_UNLOCK_OBJ (kernel);
      VG_REFC_ZERO (kernel);

      POCL_ATOMIC_DEC (kernel_c);

      TP_FREE_KERNEL (kernel->context->id, kernel->id, kernel->name);

      POCL_MSG_PRINT_REFCOUNTS ("Free Kernel %s (%p)\n", kernel->name, kernel);
      cl_program program = kernel->program;
      assert (program != NULL);

      /* Find the kernel in the program's linked list of kernels */
      POCL_LOCK_OBJ (program);
      LL_DELETE (program->kernels, kernel);

      for (i = 0; i < program->num_devices; ++i)
        {
          cl_device_id device = program->devices[i];
          if (device->ops->free_kernel && (*(device->available) == CL_TRUE))
            device->ops->free_kernel (device, program, kernel, i);
        }

      if (kernel->meta->total_argument_storage_size)
        {
          POCL_MEM_FREE (kernel->dyn_argument_storage);
          POCL_MEM_FREE (kernel->dyn_argument_offsets);
        }
      else
        {
          for (i = 0; i < (kernel->meta->num_args); i++)
            {
              pocl_aligned_free (kernel->dyn_arguments[i].value);
            }
        }
      kernel->name = NULL;
      kernel->meta = NULL;
      POCL_MEM_FREE (kernel->data);
      POCL_MEM_FREE (kernel->dyn_arguments);
      POCL_DESTROY_OBJECT (kernel);
      POCL_MEM_FREE (kernel);
      POCL_UNLOCK_OBJ (program);

      POname(clReleaseProgram) (program);
    }
  else
    {
      VG_REFC_NONZERO (kernel);
      POCL_UNLOCK_OBJ (kernel);
    }

  return CL_SUCCESS;
}
POsym(clReleaseKernel)
