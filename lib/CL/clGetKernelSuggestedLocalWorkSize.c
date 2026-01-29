/* OpenCL runtime library: clGetDeviceInfo()

   Copyright (c) 2026 Michal Babej / Tampere University

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

#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clGetKernelSuggestedLocalWorkSizeKHR) (
  cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  size_t *suggested_local_work_size)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND ((suggested_local_work_size == NULL),
                          CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON (
    (command_queue->context != kernel->context), CL_INVALID_CONTEXT,
    "memobj and command_queue are not from the same context\n");

  cl_program program = kernel->program;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_COND ((program->build_status != CL_BUILD_SUCCESS),
                          CL_INVALID_PROGRAM_EXECUTABLE);

  for (cl_uint i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument *p = &kernel->dyn_arguments[i];
      POCL_RETURN_ERROR_ON ((!p->is_set), CL_INVALID_KERNEL_ARGS,
                            "%u -th kernel argument is not set\n", i);
    }

  POCL_RETURN_ERROR_COND ((work_dim == 0), CL_INVALID_WORK_DIMENSION);
  POCL_RETURN_ERROR_COND ((work_dim > 3), CL_INVALID_WORK_DIMENSION);

  POCL_RETURN_ERROR_COND ((global_work_size == NULL),
                          CL_INVALID_GLOBAL_WORK_SIZE);
  for (cl_uint i = 0; i < work_dim; ++i)
    {
      POCL_RETURN_ERROR_COND ((global_work_size[i] == 0),
                              CL_INVALID_GLOBAL_WORK_SIZE);
      if (global_work_offset)
        {
          size_t max = SIZE_MAX - global_work_size[i];
          POCL_RETURN_ERROR_COND ((global_work_offset[i] > max),
                                  CL_INVALID_GLOBAL_OFFSET);
        }
    }

  cl_uint program_dev_i = CL_UINT_MAX;
  cl_device_id realdev = pocl_real_dev (command_queue->device);
  for (unsigned i = 0; i < kernel->program->num_devices; ++i)
    {
      if (kernel->program->devices[i] == realdev)
        program_dev_i = i;
    }
  assert (program_dev_i < CL_UINT_MAX);

  size_t offset[3];
  size_t num_groups[3];
  return pocl_kernel_calc_wg_size (
    realdev, kernel, program_dev_i, work_dim, global_work_offset,
    global_work_size, NULL, offset, suggested_local_work_size, num_groups);
}
POsym (clGetKernelSuggestedLocalWorkSizeKHR)
