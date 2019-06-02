/* OpenCL runtime library: clCreateSampler()

   Copyright (c) 2012-2017 pocl developers

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
#include "pocl_icd.h"
#include "pocl_util.h"

static unsigned long sampler_ids = 0;

extern CL_API_ENTRY cl_sampler CL_API_CALL
POname(clCreateSampler)(cl_context          context,
                cl_bool             normalized_coords, 
                cl_addressing_mode  addressing_mode, 
                cl_filter_mode      filter_mode,
                cl_int *            errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  int errcode = CL_SUCCESS;
  cl_sampler sampler = NULL;

  POCL_GOTO_ERROR_COND ((context == NULL), CL_INVALID_CONTEXT);

  /* at least 1 device must support images */
  size_t i, any_device_has_images = 0;
  for (i = 0; i < context->num_devices; i++)
    any_device_has_images += (size_t)context->devices[i]->image_support;
  POCL_GOTO_ERROR_ON ((!any_device_has_images), CL_INVALID_OPERATION,
                      "None of the devices within context support images\n");

  /* check requested sampler validity */
  POCL_GOTO_ERROR_COND (
      ((normalized_coords != CL_TRUE) && (normalized_coords != CL_FALSE)),
      CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND (((normalized_coords != CL_TRUE)
                         && (addressing_mode == CL_ADDRESS_MIRRORED_REPEAT)),
                        CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND (((normalized_coords != CL_TRUE)
                         && (addressing_mode == CL_ADDRESS_REPEAT)),
                        CL_INVALID_VALUE);

  sampler = (cl_sampler) malloc(sizeof(struct _cl_sampler));
  POCL_GOTO_ERROR_COND ((sampler == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT (sampler);
  POname (clRetainContext) (context);
  sampler->context = context;
  sampler->id = ATOMIC_INC (sampler_ids);
  sampler->normalized_coords = normalized_coords;
  sampler->addressing_mode = addressing_mode;
  sampler->filter_mode = filter_mode;
  sampler->device_data = (void **)calloc (pocl_num_devices, sizeof (void *));
  for (i = 0; i < context->num_devices; ++i)
    {
      cl_device_id dev = context->devices[i];
      if (dev->image_support == CL_TRUE && dev->ops->create_sampler)
        sampler->device_data[dev->dev_id]
            = dev->ops->create_sampler (dev->data, sampler, &errcode);
    }

ERROR:
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return sampler;
}
POsym(clCreateSampler)
