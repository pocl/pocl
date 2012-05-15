/* OpenCL runtime library: clSetKernelArg()

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
#include <assert.h>
#include <string.h>

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel kernel,
               cl_uint arg_index,
               size_t arg_size,
               const void *arg_value) CL_API_SUFFIX__VERSION_1_0
{
  struct pocl_argument *p;
  void *value;
  
  if (kernel == NULL)
    return CL_INVALID_KERNEL;

  if (arg_index >= kernel->num_args)
    return CL_INVALID_ARG_INDEX;
  
  if (kernel->arguments == NULL)
    return CL_INVALID_KERNEL;

  p = &(kernel->arguments[arg_index]);
  
  if (arg_value != NULL)
    {
      free (p->value);

      value = malloc (arg_size);
      if (value == NULL)
        return CL_OUT_OF_HOST_MEMORY;
      
      memcpy (value, arg_value, arg_size);

      p->value = value;
    }
  else
    {
      free (p->value);
      p->value = NULL;
    }

  p->size = arg_size;

  return CL_SUCCESS;
}
