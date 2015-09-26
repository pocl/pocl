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
#include "pocl_mem_management.h"
#include <stdlib.h>
#include <string.h>

int context_set_properties(cl_context                    context,
                           const cl_context_properties * properties,
                           cl_int *                      errcode)
{
  unsigned i;
  int num_properties = 0;
  
  context->properties = NULL;
  
  /* verify if data in properties is valid
   * and set them */
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
          /* redefinition of the same property */
          for(q=properties; q<p; q+=2)
            if (q[0] == p[0])
              {
                POCL_MSG_ERR("Duplicate properties: %lu\n", (unsigned long)q[0]);
                *errcode = CL_INVALID_PROPERTY; 
                return 0;
              }
          
          switch (p[0])
            {
            case CL_CONTEXT_PLATFORM:

              /* pocl just have one platform */
              platform_found = CL_FALSE;
              for (i=0; i<num_platforms; i++)
                if ((cl_platform_id)p[1] == platforms[i])
                  platform_found = CL_TRUE;

              if (platform_found == CL_FALSE)
                {
                  POCL_MSG_ERR("Could not find platform %p\n", (void*)p[1]);
                  *errcode = CL_INVALID_PLATFORM;
                  return 0;
                }

              p += 2;
              break;

            default: 
              POCL_MSG_ERR("Unknown context property: %lu\n", (unsigned long)p[0]);
              *errcode = CL_INVALID_PROPERTY;
              return 0;
            }
          num_properties++;
        }

      context->properties = (cl_context_properties *) malloc
        ((num_properties * 2 + 1) * sizeof(cl_context_properties));
      if (context->properties == NULL)
        {
          *errcode = CL_OUT_OF_HOST_MEMORY;
          return 0;
        }
      
      memcpy(context->properties, properties, 
             (num_properties * 2 + 1) * sizeof(cl_context_properties));
      context->num_properties = num_properties;

      *errcode = 0;
      return num_properties;
    }
  else
    {
      context->properties     = NULL;
      context->num_properties = 0;
      
      *errcode = 0;
      return 0;
    }
}

CL_API_ENTRY cl_context CL_API_CALL
POname(clCreateContext)(const cl_context_properties * properties,
                cl_uint                       num_devices,
                const cl_device_id *          devices,
                void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
                void *                        user_data,
                cl_int *                      errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i, j;
  cl_device_id device_ptr;
  int errcode = 0;
  cl_context context = NULL;

  POCL_GOTO_ERROR_COND((devices == NULL || num_devices == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((pfn_notify == NULL && user_data != NULL), CL_INVALID_VALUE);

  
  lt_dlinit();
  pocl_init_devices();

  context = (cl_context) malloc(sizeof(struct _cl_context));
  if (context == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  POCL_INIT_OBJECT(context);

  context_set_properties(context, properties, &errcode);
  if (errcode)
    {
      goto ERROR;
    }
  
  context->num_devices = num_devices;
  context->devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
  if (context->devices == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
  
  j = 0;
  for (i = 0; i < num_devices; ++i)
    {
      device_ptr = devices[i];
      if (device_ptr == NULL)
        {
          POCL_MSG_ERR("one of the devices in device list is NULL\n");
          errcode = CL_INVALID_DEVICE;
          goto ERROR_CLEAN_CONTEXT_AND_DEVICES;
        }
      
      if (device_ptr->available == CL_TRUE) 
        {
          context->devices[j] = device_ptr;
          ++j;
        }
      else
        POCL_MSG_WARN("device not available: %s\n", device_ptr->long_name);
      POname(clRetainDevice)(device_ptr);
    }

  pocl_init_mem_manager ();
  
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  context->valid = 1;
  return context;
  
ERROR_CLEAN_CONTEXT_AND_DEVICES:
  POCL_MEM_FREE(context->devices);
 /*ERROR_CLEAN_CONTEXT_AND_PROPERTIES:*/
  POCL_MEM_FREE(context->properties);
ERROR:
  POCL_MEM_FREE(context);
  if(errcode_ret != NULL)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym(clCreateContext)
