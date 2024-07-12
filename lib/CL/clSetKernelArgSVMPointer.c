/* OpenCL runtime library: clSetKernelArgSVMPointer()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "config.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "devices.h"

/**
 * Sets a raw pointer argument.
 *
 * Shared implementation helper between SVM, USM and the buffer device
 * address extension.
 */
int
pocl_set_kernel_arg_pointer (cl_kernel kernel, cl_uint arg_index,
                             const void *arg_value)
{
  POCL_RETURN_ERROR_ON ((kernel->dyn_arguments == NULL), CL_INVALID_KERNEL,
                        "This kernel has no arguments that could be set\n");

  POCL_MSG_PRINT_INFO ("Setting kernel arg %i to pointer: %p\n", arg_index,
                       arg_value);

  struct pocl_argument *p;
  struct pocl_argument_info *pi;
  POCL_RETURN_ERROR_ON ((arg_index >= kernel->meta->num_args),
                        CL_INVALID_ARG_INDEX,
                        "This kernel has %u args, cannot set arg %u\n",
                        (unsigned)kernel->meta->num_args, (unsigned)arg_index);

  p = &(kernel->dyn_arguments[arg_index]);
  pi = &(kernel->meta->arg_info[arg_index]);
  POCL_RETURN_ERROR_ON ((ARGP_IS_LOCAL (pi)), CL_INVALID_ARG_VALUE,
                        "arg %u is in local address space\n", arg_index);

  POCL_RETURN_ERROR_ON ((pi->type != POCL_ARG_TYPE_POINTER),
                        CL_INVALID_ARG_VALUE, "arg %u is not a pointer\n",
                        arg_index);

  if (kernel->dyn_argument_storage != NULL)
    p->value = kernel->dyn_argument_offsets[arg_index];
  else if (p->value == NULL)
    {
      p->value = pocl_aligned_malloc (sizeof (void *), sizeof (void *));
      POCL_RETURN_ERROR_COND ((p->value == NULL), CL_OUT_OF_HOST_MEMORY);
    }
  memcpy (p->value, &arg_value, sizeof (void *));

  p->is_set = 1;
  p->is_readonly = 0;
  p->is_raw_ptr = 1;
  p->size = sizeof (void *);

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clSetKernelArgSVMPointer) (cl_kernel kernel, cl_uint arg_index,
                                   const void *arg_value)
    CL_API_SUFFIX__VERSION_2_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON (
      (!kernel->context->svm_allocdev), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  return pocl_set_kernel_arg_pointer (kernel, arg_index, arg_value);
}
POsym(clSetKernelArgSVMPointer)
