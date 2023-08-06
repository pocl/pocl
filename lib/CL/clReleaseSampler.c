/* OpenCL runtime library: clReleaseSampler()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#include "pocl_util.h"

extern unsigned long sampler_c;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseSampler)(cl_sampler sampler)
CL_API_SUFFIX__VERSION_1_0
{
  unsigned i;
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (sampler)), CL_INVALID_SAMPLER);

  int new_refcount;
  POCL_RELEASE_OBJECT (sampler, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Sampler %" PRId64 " (%p), Refcount: %d\n",
                            sampler->id, sampler, new_refcount);

  if (new_refcount == 0)
    {
      VG_REFC_ZERO (sampler);
      POCL_ATOMIC_DEC (sampler_c);

      POCL_MSG_PRINT_REFCOUNTS ("Free Sampler %" PRId64 " (%p)\n", sampler->id,
                                sampler);

      cl_context context = sampler->context;
      TP_FREE_SAMPLER (context->id, sampler->id);
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          if (*(dev->available) == CL_FALSE)
            continue;
          if (dev->image_support == CL_TRUE && dev->ops->free_sampler)
            {
              dev->ops->free_sampler (dev, sampler, dev->dev_id);
              sampler->device_data[dev->dev_id] = NULL;
            }
        }
      POCL_MEM_FREE (sampler->device_data);
      POCL_DESTROY_OBJECT (sampler);
      POCL_MEM_FREE (sampler);
      POname (clReleaseContext) (context);
    }
  else
    {
      VG_REFC_NONZERO (sampler);
    }

  return CL_SUCCESS;
}
POsym(clReleaseSampler)
