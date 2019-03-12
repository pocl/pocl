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
#include "pocl_util.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include <stdlib.h>
#include <string.h>

extern unsigned cl_context_count;
extern pocl_lock_t pocl_context_handling_lock;

/* in clCreateContext.c */
int context_set_properties(cl_context                    ctx,
                           const cl_context_properties * properties,
                           cl_int *                      errcode_ret);

CL_API_ENTRY cl_context CL_API_CALL
POname(clCreateContextFromType)(const cl_context_properties *properties,
                        cl_device_type device_type,
                        void (*pfn_notify)(const char *, const void *, size_t, void *),
                        void *user_data,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_uint i, num_devices;
  cl_context context = NULL;
  int errcode;
  cl_device_id device_ptr;

  POCL_LOCK (pocl_context_handling_lock);

  POCL_GOTO_ERROR_COND((pfn_notify == NULL && user_data != NULL), CL_INVALID_VALUE);

  /* initialize libtool here, LT will be needed when loading the kernels */     
  lt_dlinit();
  errcode = pocl_init_devices();
  /* see clCreateContext.c for explanation */
  POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), errcode,
                      "Could not initialize devices\n");

  context = (cl_context)calloc (1, sizeof (struct _cl_context));
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

  num_devices = pocl_get_device_type_count(device_type);

  if (num_devices == 0)
    {
      if (errcode_ret != NULL) 
        {
          *errcode_ret = (CL_DEVICE_NOT_FOUND); 
        } 
      /* Return a dummy context so icd call to clReleaseContext() still
         works. This fixes AMD SDK OpenCL samples to work (as of 2012-12-05). */
      POCL_MSG_WARN("Couldn't find any device of type %lu; returning "
                    "a dummy context with 0 devices\n", (unsigned long)device_type);
      POCL_UNLOCK (pocl_context_handling_lock);
      return context;
    }

  context->num_devices = num_devices;
  context->devices = (cl_device_id *) calloc(num_devices, sizeof(cl_device_id));
  if (context->devices == NULL)
    {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_CONTEXT_AND_PROPERTIES;
    }

  pocl_get_devices(device_type, context->devices, num_devices);

  for (i = 0; i < num_devices; ++i)
    {
      device_ptr = context->devices[i];
      if (device_ptr == NULL)
        {
          break;
        }

      POname(clRetainDevice)(device_ptr);
    }

  pocl_setup_context(context);

  pocl_init_mem_manager ();

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  context->valid = 1;

  cl_context_count += 1;
  POCL_UNLOCK (pocl_context_handling_lock);

  return context;

ERROR_CLEAN_CONTEXT_AND_PROPERTIES:
  POCL_MEM_FREE (context->devices);
  POCL_MEM_FREE(context->properties);
ERROR:
  POCL_MEM_FREE (context);
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  POCL_UNLOCK (pocl_context_handling_lock);
  return NULL;
}
POsym(clCreateContextFromType)
