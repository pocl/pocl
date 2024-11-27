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

#include "devices.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"
#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

int
context_set_properties (cl_context context,
                        const cl_context_properties *properties)
{
  unsigned i;
  int num_properties = 0;
  
  context->properties = NULL;
  context->gl_interop = CL_FALSE;

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
                return CL_INVALID_PROPERTY;
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
                  return CL_INVALID_PLATFORM;
                }

              p += 2;
              break;

            case CL_CONTEXT_INTEROP_USER_SYNC:
            case CL_GL_CONTEXT_KHR:
            case CL_EGL_DISPLAY_KHR:
            case CL_GLX_DISPLAY_KHR:
            case CL_CGL_SHAREGROUP_KHR:
              context->gl_interop = CL_TRUE;
              p += 2;
              break;

            default: 
              POCL_MSG_ERR("Unknown context property: %lu\n", (unsigned long)p[0]);
              return CL_INVALID_PROPERTY;
            }
          num_properties++;
        }

      context->properties = (cl_context_properties *) malloc
        ((num_properties * 2 + 1) * sizeof(cl_context_properties));
      if (context->properties == NULL)
        {
          return CL_OUT_OF_HOST_MEMORY;
        }
      
      memcpy(context->properties, properties, 
             (num_properties * 2 + 1) * sizeof(cl_context_properties));
      context->num_properties = num_properties;

      return CL_SUCCESS;
    }
  else
    {
      context->properties     = NULL;
      context->num_properties = 0;
      
      return 0;
    }
}

extern int pocl_offline_compile;

unsigned cl_context_count = 0;
pocl_lock_t pocl_context_handling_lock;

CL_API_ENTRY cl_context CL_API_CALL
POname(clCreateContext)(const cl_context_properties * properties,
                cl_uint                       num_devices,
                const cl_device_id *          devices,
                void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
                void *                        user_data,
                cl_int *                      errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i = 0;
  cl_int errcode = 0;
  cl_context context = NULL;

  POCL_LOCK (pocl_context_handling_lock);

#ifdef ENABLE_LLVM
  InitializeLLVM ();
#endif

  POCL_GOTO_ERROR_COND((devices == NULL || num_devices == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((pfn_notify == NULL && user_data != NULL), CL_INVALID_VALUE);

  errcode = pocl_init_devices();
  /* clCreateContext cannot return CL_DEVICE_NOT_FOUND, which is what
   * pocl_init_devices() returns if no devices could be probed. Hence,
   * remap this error to CL_INVALID_DEVICE. Note that this particular
   * situation should never arise, since an application should issue
   * clGetDeviceIDs before clCreateContext, and we would have returned
   * CL_DEVICE_NOT_FOUND already from clGetDeviceIDs. Still, no reason
   * not to handle it.
   */

  if (errcode == CL_DEVICE_NOT_FOUND)
    errcode = CL_INVALID_DEVICE;
  POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), errcode,
                      "Could not initialize devices\n");

  for (i = 0; i < num_devices; ++i)
    {
      POCL_GOTO_ERROR_ON ((devices[i] == NULL), CL_INVALID_DEVICE,
                          "one of the devices in device list is NULL\n");
    }

  context = (cl_context)calloc (1, sizeof (struct _cl_context));
  POCL_GOTO_ERROR_COND ((context == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT(context);

  errcode = context_set_properties (context, properties);
  if (errcode)
    goto ERROR;

  context->create_devices = calloc(num_devices, sizeof(cl_device_id));
  POCL_GOTO_ERROR_COND ((context->create_devices == NULL), CL_OUT_OF_HOST_MEMORY);
  memcpy(context->create_devices, devices, num_devices * sizeof(cl_device_id));
  context->num_create_devices = num_devices;

  context->devices = pocl_unique_device_list(devices, num_devices,
                                             &context->num_devices);
  POCL_GOTO_ERROR_COND ((context->devices == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_GOTO_ERROR_ON ((context->num_devices == 0), CL_INVALID_DEVICE,
                      "Zero devices\n");

  context->default_queues
      = (cl_command_queue *)calloc (num_devices, sizeof (cl_command_queue));
  POCL_GOTO_ERROR_COND ((context->default_queues == NULL),
                        CL_OUT_OF_HOST_MEMORY);

  for (i = 0; i < context->num_devices; ++i)
    {
      cl_device_id dev = context->devices[i];
      POCL_GOTO_ERROR_ON (
          (!pocl_offline_compile && (*dev->available == CL_FALSE)),
          CL_INVALID_DEVICE,
          "Device unavailable and offline compilation "
          "disabled: %s\n",
          dev->long_name);
      context->mem_base_addr_align
        = max (dev->mem_base_addr_align, context->mem_base_addr_align);
    }

  pocl_init_mem_manager ();

  /* only required for online context */
  if (!pocl_offline_compile)
    {
      errcode = pocl_setup_context (context);
      if (errcode)
        goto ERROR;
    }

  for (i = 0; i < context->num_create_devices; ++i)
    POname (clRetainDevice) (context->create_devices[i]);

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;

#ifdef ENABLE_LLVM
  pocl_llvm_create_context (context);
#endif

  POCL_ATOMIC_INC (context_c);

  cl_context_count += 1;
  POCL_UNLOCK (pocl_context_handling_lock);

  POCL_MSG_PRINT_GENERAL ("Created Context %" PRId64 " (%p)\n", context->id,
                          context);
  return context;
  
ERROR:
  if (context)
    {
      for (i = 0; i < context->num_devices; i++)
        {
          if (context->default_queues && context->default_queues[i])
            {
              POname (clReleaseCommandQueue) (context->default_queues[i]);
            }
        }
      for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
        POCL_MEM_FREE (context->image_formats[i]);
      POCL_MEM_FREE (context->default_queues);
      POCL_MEM_FREE (context->devices);
      POCL_MEM_FREE (context->create_devices);
      POCL_MEM_FREE (context->properties);
    }
  POCL_MEM_FREE(context);
  if(errcode_ret != NULL)
    {
      *errcode_ret = errcode;
    }

  POCL_UNLOCK (pocl_context_handling_lock);
  return NULL;
}
POsym(clCreateContext)
