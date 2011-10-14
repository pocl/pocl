/* OpenCL runtime library: clCreateBuffer()

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

CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context,
               cl_mem_flags flags,
               size_t size,
               void *host_ptr,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem;
  cl_device_id device_id;
  void *device_ptr;
  unsigned i;

  if (context == NULL)
    POCL_ERROR(CL_INVALID_CONTEXT);

  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);
  mem->device_ptrs = (void **) malloc(context->num_devices * sizeof(void *));
  if (mem->device_ptrs == NULL)
    {
      free(mem);
      POCL_ERROR(CL_OUT_OF_HOST_MEMORY);
    }
  
  for (i = 0; i < context->num_devices; ++i)
    {
      device_id = context->devices[i];
      device_ptr = device_id->malloc(device_id->data, flags, size, host_ptr);
      if (device_ptr == NULL)
	{
	  clReleaseMemObject(mem);
	  POCL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
	}
      mem->device_ptrs[i] = device_ptr;
    }

  mem->size = size;
  mem->mem_host_ptr = host_ptr;
  mem->reference_count = 1;
  mem->context = context;
      
  return mem;
}
