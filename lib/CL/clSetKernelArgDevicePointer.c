/* OpenCL runtime library: clSetKernelArgSVMPointer()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "config.h"
#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

/** Sets a raw device pointer as a buffer argument.
 *
 * Part of the cl_ext_buffer_device_address extension.
 */
CL_API_ENTRY cl_int CL_API_CALL
POname (clSetKernelArgDevicePointerEXT) (cl_kernel kernel, cl_uint arg_index,
                                         cl_mem_device_address_EXT dev_addr)
    CL_API_SUFFIX__VERSION_1_2
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  int found_supported_dev = 0;
  for (size_t i = 0; i < kernel->context->num_devices; ++i)
    {
      cl_device_id dev = kernel->context->devices[i];
      if (strstr("cl_ext_buffer_device_address", dev->extensions) == 0)
        {
          found_supported_dev = 1;
          break;
        }
    }

  POCL_RETURN_ERROR_ON (
      (!found_supported_dev), CL_INVALID_OPERATION,
      "None of the devices in this context supports 'cl_ext_buffer_device_address'\n");

  return pocl_set_kernel_arg_pointer (kernel, arg_index, (void *)dev_addr);
}
POsym(clSetKernelArgDevicePointerEXT)
