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

#include "locl_cl.h"
#include <assert.h>
#include <string.h>

#define ARGUMENT_STRING_LENGTH 32

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel kernel,
               cl_uint arg_index,
               size_t arg_size,
               const void *arg_value) CL_API_SUFFIX__VERSION_1_0
{
  char s[ARGUMENT_STRING_LENGTH];
  int error;
  void *p;

  if (kernel == NULL)
    return CL_INVALID_KERNEL;

  if (arg_index >= kernel->num_args)
    return CL_INVALID_ARG_INDEX;

  error = snprintf(s, ARGUMENT_STRING_LENGTH,
		   "_arg%u", arg_index);
  assert(error > 0);
  
  p = lt_dlsym(kernel->dlhandle, s);
  if (p == NULL)
    return CL_INVALID_KERNEL;

  memcpy(p, arg_value, arg_size);

  error = snprintf(s, ARGUMENT_STRING_LENGTH,
		   "_size%u", arg_index);
  assert(error > 0);
  
  p = lt_dlsym(kernel->dlhandle, s);
  if (p == NULL)
    return CL_INVALID_KERNEL;

  *(size_t *) p = arg_size;

  return CL_SUCCESS;
}
