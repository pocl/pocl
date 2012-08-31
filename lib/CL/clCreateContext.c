/* OpenCL runtime library: clCreateContext()

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

#include "devices/devices.h"
#include "pocl_cl.h"
#include "pocl_icd.h"
#include <stdlib.h>

CL_API_ENTRY cl_context CL_API_CALL
POclCreateContext(const cl_context_properties * properties,
                cl_uint                       num_devices,
                const cl_device_id *          devices,
                void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
                void *                        user_data,
                cl_int *                      errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  int i, j;
  cl_device_id device_ptr;

  lt_dlinit();
  pocl_init_devices();

  cl_context context = (cl_context) malloc(sizeof(struct _cl_context));
  if (context == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT(context);

  context->num_devices = num_devices;
  context->devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
  POCL_INIT_ICD_OBJECT(context);
  
  j = 0;
  for (i = 0; i < num_devices; ++i) 
    {
      device_ptr = devices[i];
      if (device_ptr == NULL)
        POCL_ERROR(CL_INVALID_DEVICE);

      if (device_ptr->available == CL_TRUE) 
        {
          context->devices[j] = device_ptr;
          ++j;
        }
      POclRetainDevice(device_ptr);
    }   

  context->properties = properties;

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return context;
}
POsym(clCreateContext)
