/* OpenCL runtime library: clGetKernelSubGroupInfo()

   Copyright (c) 2021 Väinö Liukko

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_ENTRY POname (clGetKernelSubGroupInfo) (
    cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_2_1
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  /* Check that kernel is associated with device, or that there is no
     risk of confusion. */
  if (device != NULL)
    {
      unsigned i;
      int found_it = 0;
      for (i = 0; i < kernel->context->num_devices; i++)
        if (pocl_real_dev (device) == kernel->context->devices[i])
          {
            found_it = 1;
            break;
          }
      POCL_RETURN_ERROR_ON ((!found_it), CL_INVALID_DEVICE,
                            "could not find the "
                            "device supplied in argument\n");
    }
  else
    {
      POCL_RETURN_ERROR_ON ((kernel->context->num_devices > 1),
                            CL_INVALID_DEVICE,
                            "No device given and context has > 1 device\n");
      device = kernel->context->devices[0];
    }

  /* Check device for subgroup support */
  if (device->max_num_sub_groups == 0)
    {
      return CL_INVALID_OPERATION;
    }

  /* In case device reports subgroup support */
  POCL_ABORT_UNIMPLEMENTED (
      "device associated with the kernel falsely indicates "
      "subgroup support\n");
  return CL_INVALID_OPERATION;
}
