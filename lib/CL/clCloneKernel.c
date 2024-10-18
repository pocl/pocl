/* OpenCL runtime library: clCloneKernel()

   Copyright (c) 2022 Michal Babej / Tampere University

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

#include "pocl_binary.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_util.h"
#include "utlist.h"

CL_API_ENTRY cl_kernel CL_API_CALL
POname (clCloneKernel) (cl_kernel source_kernel,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_1
{
  cl_kernel kernel = NULL;
  int errcode = CL_SUCCESS;
  size_t i;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (source_kernel)),
                        CL_INVALID_KERNEL);

  cl_program program = source_kernel->program;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  POCL_GOTO_ERROR_ON ((program->build_status == CL_BUILD_NONE),
                      CL_INVALID_PROGRAM_EXECUTABLE,
                      "You must call clBuildProgram first!"
                      " (even for programs created with binaries)\n");

  POCL_GOTO_ERROR_ON ((program->build_status != CL_BUILD_SUCCESS),
                      CL_INVALID_PROGRAM_EXECUTABLE,
                      "Last BuildProgram() was not successful\n");

  assert (program->num_devices != 0);

  kernel = (cl_kernel)calloc (1, sizeof (struct _cl_kernel));
  POCL_GOTO_ERROR_ON ((kernel == NULL), CL_OUT_OF_HOST_MEMORY,
                      "clCloneKernel couldn't allocate memory");
  POCL_INIT_OBJECT (kernel);

  kernel->meta = source_kernel->meta;
  kernel->data = (void **)calloc (program->num_devices, sizeof (void *));
  kernel->name = source_kernel->meta->name;
  kernel->context = program->context;
  kernel->program = program;

  kernel->dyn_arguments = (pocl_argument *)calloc (
      (kernel->meta->num_args), sizeof (struct pocl_argument));
  POCL_GOTO_ERROR_COND ((kernel->dyn_arguments == NULL),
                        CL_OUT_OF_HOST_MEMORY);
  /* this memcpy copies the metadata ('is_set' etc),
   * but we must still properly set the "value" pointer. */
  memcpy (kernel->dyn_arguments, source_kernel->dyn_arguments,
          (kernel->meta->num_args) * sizeof (struct pocl_argument));

  if (kernel->meta->total_argument_storage_size)
    {
      kernel->dyn_argument_storage
          = (char *)calloc (1, kernel->meta->total_argument_storage_size);
      kernel->dyn_argument_offsets
          = (void **)malloc (kernel->meta->num_args * sizeof (void *));

      /* copy the content of arguments */
      memcpy (kernel->dyn_argument_storage,
              source_kernel->dyn_argument_storage,
              kernel->meta->total_argument_storage_size);

      /* offsets must be recalculated */
      size_t offset = 0;
      for (i = 0; i < kernel->meta->num_args; ++i)
        {
          kernel->dyn_argument_offsets[i]
              = kernel->dyn_argument_storage + offset;
          unsigned type_size = kernel->meta->arg_info[i].type_size;
          assert (type_size > 0);
          offset += type_size;
          kernel->dyn_arguments[i].value = kernel->dyn_argument_offsets[i];
        }
      assert (offset == kernel->meta->total_argument_storage_size);
    }
  else
    {
      /* clone arguments one by one */
      for (i = 0; i < kernel->meta->num_args; ++i)
        {
          struct pocl_argument *p = &kernel->dyn_arguments[i];
          struct pocl_argument *sp = &source_kernel->dyn_arguments[i];
          struct pocl_argument_info *pi = &(kernel->meta->arg_info[i]);
          if (p->is_set && sp->value != NULL) /* local args can have NULL */
            {
              size_t arg_alignment, arg_alloc_size;
              arg_alignment = pocl_size_ceil2 (p->size);
              if (arg_alignment >= MAX_EXTENDED_ALIGNMENT)
                arg_alignment = MAX_EXTENDED_ALIGNMENT;

              arg_alloc_size = p->size;
              if (arg_alloc_size < arg_alignment)
                arg_alloc_size = arg_alignment;

              assert (arg_alloc_size > 0);
              p->value = pocl_aligned_malloc (arg_alignment, arg_alloc_size);
              memcpy (p->value, sp->value, p->size);
            }
          else
            p->value = NULL;
        }
    }

  for (i = 0; i < program->num_devices; ++i)
    {
      cl_device_id device = program->devices[i];
      if (device->ops->create_kernel)
        {
          int r = device->ops->create_kernel (device, program, kernel, i);
          POCL_GOTO_ERROR_ON ((r != CL_SUCCESS), CL_OUT_OF_RESOURCES,
                              "could not create device-specific data "
                              "for kernel %s\n",
                              kernel->name);
        }
    }

  TP_CREATE_KERNEL (kernel->context->id, kernel->id, kernel->name);

  POCL_LOCK_OBJ (program);
  LL_PREPEND (program->kernels, kernel);
  POCL_RETAIN_OBJECT_UNLOCKED (program);
  POCL_UNLOCK_OBJ (program);

  POCL_ATOMIC_INC (kernel_c);

  errcode = CL_SUCCESS;
  goto SUCCESS;

ERROR:
  if (kernel)
    {
      POCL_MEM_FREE (kernel->dyn_arguments);
      POCL_MEM_FREE (kernel->data);
      if (kernel->meta->total_argument_storage_size)
        {
          POCL_MEM_FREE (kernel->dyn_argument_storage);
          POCL_MEM_FREE (kernel->dyn_argument_offsets);
        }
      else
        {
          for (i = 0; i < kernel->meta->num_args; ++i)
            {
              struct pocl_argument *p = &kernel->dyn_arguments[i];
              pocl_aligned_free (p->value);
            }
        }
    }
  POCL_MEM_FREE (kernel);
  kernel = NULL;

SUCCESS:
  if (errcode_ret != NULL)
    {
      *errcode_ret = errcode;
    }
  return kernel;
}
POsym (clCloneKernel)
