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
#include "pocl_util.h"

extern unsigned long sampler_c;

CL_API_ENTRY cl_sampler CL_API_CALL
POname(clCreateSampler)(cl_context          context,
                cl_bool             normalized_coords, 
                cl_addressing_mode  addressing_mode, 
                cl_filter_mode      filter_mode,
                cl_int *            errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  int errcode = CL_SUCCESS;
  cl_sampler sampler = NULL;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

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

  sampler = (cl_sampler)calloc (1, sizeof (struct _cl_sampler));
  POCL_GOTO_ERROR_COND ((sampler == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT (sampler);
  POname (clRetainContext) (context);
  sampler->context = context;
  sampler->normalized_coords = normalized_coords;
  sampler->addressing_mode = addressing_mode;
  sampler->filter_mode = filter_mode;
  sampler->device_data
      = (void **)calloc (POCL_ATOMIC_LOAD (pocl_num_devices), sizeof (void *));

  TP_CREATE_SAMPLER (context->id, sampler->id);

  POCL_ATOMIC_INC (sampler_c);

  for (i = 0; i < context->num_devices; ++i)
    {
      cl_device_id dev = context->devices[i];
      if (*(dev->available) == CL_FALSE)
        continue;
      if (dev->image_support == CL_TRUE && dev->ops->create_sampler)
        dev->ops->create_sampler (dev, sampler, dev->dev_id);
    }

ERROR:
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return sampler;
}
POsym (clCreateSampler)



CL_API_ENTRY cl_sampler CL_API_CALL
POname (clCreateSamplerWithProperties) (
        cl_context context, const cl_sampler_properties *sampler_properties,
        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{

  cl_bool normalized_coords = CL_TRUE;
  cl_addressing_mode addressing_mode = CL_ADDRESS_CLAMP;
  cl_filter_mode filter_mode = CL_FILTER_NEAREST;
  int coords_set = 0, addr_set = 0, filter_set = 0;
  int errcode;

  POCL_GOTO_ERROR_COND ((sampler_properties == NULL), CL_INVALID_VALUE);

  const cl_sampler_properties *p = sampler_properties;
  while (*p != 0)
    {
      switch (*p)
        {
        case CL_SAMPLER_NORMALIZED_COORDS:
          {
            POCL_GOTO_ERROR_ON ((coords_set != 0), CL_INVALID_VALUE,
                                "CL_SAMPLER_NORMALIZED_COORDS property "
                                "has been already set");
            normalized_coords = p[1];
            coords_set = 1;
            break;
          }
        case CL_SAMPLER_ADDRESSING_MODE:
          {
            POCL_GOTO_ERROR_ON ((addr_set != 0), CL_INVALID_VALUE,
                                "CL_SAMPLER_ADDRESSING_MODE property "
                                "has been already set");
            addressing_mode = p[1];
            addr_set = 1;
            break;
          }
        case CL_SAMPLER_FILTER_MODE:
          {
            POCL_GOTO_ERROR_ON ((filter_set != 0), CL_INVALID_VALUE,
                                "CL_SAMPLER_FILTER_MODE property "
                                "has been already set");
            filter_mode = p[1];
            filter_set = 1;
            break;
          }
        default:
          POCL_GOTO_ERROR_ON (1, CL_INVALID_VALUE,
                              "Unknown value in properties: %lu\n",
                              (unsigned long)(*p));
        }
      p += 2;
    }
  unsigned num_props = (p - sampler_properties) + 1; /* include final 0 */
  cl_sampler ret_sam = POname (clCreateSampler) (
      context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
  if (ret_sam == NULL)
    return NULL;

  ret_sam->num_properties = num_props;
  assert (num_props < 10);
  memcpy (ret_sam->properties, sampler_properties,
          num_props * sizeof (cl_sampler_properties));

  return ret_sam;

ERROR:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }

  return NULL;
}
POsym (clCreateSamplerWithProperties)
