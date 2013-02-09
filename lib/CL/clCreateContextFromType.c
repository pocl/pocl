/* OpenCL runtime library: clCreateContextFromType()

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
#include <stdlib.h>
#include <string.h>

CL_API_ENTRY cl_context CL_API_CALL
POname(clCreateContextFromType)(const cl_context_properties *properties,
                        cl_device_type device_type,
                        void (*pfn_notify)(const char *, const void *, size_t, void *),
                        void *user_data,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  int num_devices;
  int i, j;
  int num_properties;

  if (properties)
    {
      const cl_context_properties *p = properties;
      const cl_context_properties *q;

      cl_platform_id platforms[1];
      cl_uint num_platforms;
      cl_bool platform_found;

      POname(clGetPlatformIDs)(1, platforms, &num_platforms);

      num_properties = 0;
      while (p[0] != 0)
        {
          for(q=properties; q<p; q+=2)
            if (q[0] == p[0])
              POCL_ERROR(CL_INVALID_PROPERTY);

          switch (p[0])
            {
              case CL_CONTEXT_PLATFORM:

                platform_found = CL_FALSE;
                for (i=0; i<num_platforms; i++)
                  if ((cl_platform_id)p[1] == platforms[i])
                    platform_found = CL_TRUE;

                if (platform_found == CL_FALSE)
                  POCL_ERROR(CL_INVALID_PROPERTY);

                p += 2;
                break;

              default: POCL_ERROR(CL_INVALID_PROPERTY);
            }
          num_properties++;
        }
    }

  /* initialize libtool here, LT will be needed when loading the kernels */     
  lt_dlinit();
  pocl_init_devices();

  cl_context context = (cl_context) malloc(sizeof(struct _cl_context));
  if (context == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT(context);
  context->valid = 0;

  num_devices = 0;
  for (i = 0; i < pocl_num_devices; ++i) {
    if ((pocl_devices[i].type & device_type) &&
        (pocl_devices[i].available == CL_TRUE))
      ++num_devices;
  }

  if (num_devices == 0)
    {
      if (errcode_ret != NULL) 
        {
          *errcode_ret = (CL_DEVICE_NOT_FOUND); 
        } 
      /* Return a dummy context so icd call to clReleaseContext() still
         works. This fixes AMD SDK OpenCL samples to work (as of 2012-12-05). */
      return context;
    }

  context->num_devices = num_devices;
  context->devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
  
  j = 0;
  for (i = 0; i < pocl_num_devices; ++i) {
    if ((pocl_devices[i].type & device_type) &&
	(pocl_devices[i].available == CL_TRUE)) {
      context->devices[j] = &pocl_devices[i];
      POname(clRetainDevice)(&pocl_devices[i]);
      ++j;
    }
  }   

  context->properties = (cl_context_properties *) malloc((num_properties * 2 + 1) * sizeof(cl_context_properties));
  memcpy(context->properties, properties, (num_properties * 2 + 1) * sizeof(cl_context_properties));
  context->num_properties = num_properties;

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  context->valid = 1;
  return context;
}
POsym(clCreateContextFromType)
