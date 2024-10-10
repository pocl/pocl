/* OpenCL runtime library: clEnqueueSVMMemFill()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

cl_int
pocl_svm_memfill_common (cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_command_type command_type,
                         void *svm_ptr,
                         size_t size,
                         const void *pattern,
                         size_t pattern_size,
                         cl_uint num_items_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event,
                         const cl_sync_point_khr *sync_point_wait_list,
                         cl_sync_point_khr *sync_point,
                         _cl_command_node **cmd)
{
  POCL_VALIDATE_WAIT_LIST_PARAMS;

  unsigned i;
  cl_device_id device;

  POCL_CHECK_DEV_IN_CMDQ;
  cl_int errcode;

  /* command_queue can be NULL when pushing to a command buffer. */
  if (command_queue != NULL)
    {
      POCL_CHECK_DEV_IN_CMDQ;
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                              CL_INVALID_COMMAND_QUEUE);
    }

  cl_context context = command_queue->context;

  POCL_RETURN_ERROR_ON (
      (context->svm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND((svm_ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND((pattern_size == 0), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND((pattern_size > 128), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((__builtin_popcount(pattern_size) > 1), CL_INVALID_VALUE,
                       "pattern_size (%zu) must be a power-of-2 value\n", pattern_size);

  POCL_RETURN_ERROR_COND((size == 0), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON(((intptr_t)svm_ptr % pattern_size > 0), CL_INVALID_VALUE,
                       "svm_ptr must be aligned to pattern_size\n");

  POCL_RETURN_ERROR_ON((size % pattern_size > 0), CL_INVALID_VALUE,
                       "size must be a multiple of pattern_size\n");

  errcode = pocl_svm_check_pointer (context, svm_ptr, size, NULL);
  if (errcode != CL_SUCCESS)
    return errcode;

  /* Utilize the SVM shadow buffers to share code with cl_mem buffer fill
     code. */

  pocl_raw_ptr *dst_svm_ptr = pocl_find_raw_ptr_with_vm_ptr (context, svm_ptr);

  size_t offset = (char *)svm_ptr - (char *)dst_svm_ptr->vm_ptr;
  if (command_buffer)
    errcode = POname (clCommandFillBufferKHR) (
        command_buffer, NULL, NULL, dst_svm_ptr->shadow_cl_mem, pattern,
        pattern_size, offset, size, num_items_in_wait_list,
        sync_point_wait_list, sync_point, NULL);
  else
    errcode = POname (clEnqueueFillBuffer) (
        command_queue, dst_svm_ptr->shadow_cl_mem, pattern, pattern_size,
        offset, size, num_items_in_wait_list, event_wait_list, event);

  if (errcode != CL_SUCCESS)
    return errcode;

  // TODO this is likely very wrong
  if (event != NULL)
    (*event)->command_type = command_type;

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueSVMMemFill) (cl_command_queue command_queue, void *svm_ptr,
                              const void *pattern, size_t pattern_size,
                              size_t size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event) CL_API_SUFFIX__VERSION_2_0
{
  return pocl_svm_memfill_common (NULL, command_queue, CL_COMMAND_SVM_MEMFILL,
                                  svm_ptr, size, pattern, pattern_size,
                                  num_events_in_wait_list, event_wait_list,
                                  event, NULL, NULL, NULL);
}
POsym(clEnqueueSVMMemFill)
