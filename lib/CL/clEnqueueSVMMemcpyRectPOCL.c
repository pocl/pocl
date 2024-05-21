/* OpenCL runtime library: clEnqueueSVMMemcpyRectPOCL()

   Copyright (c) 2023-2024 Michal Babej / Intel Finland Oy

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
pocl_svm_memcpy_rect_common (cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             void *dst_ptr,
                             const void *src_ptr,
                             const size_t *dst_origin,
                             const size_t *src_origin,
                             const size_t *region,
                             size_t dst_row_pitch,
                             size_t dst_slice_pitch,
                             size_t src_row_pitch,
                             size_t src_slice_pitch,
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
  const size_t zero_origin[3] = { 0, 0, 0 };

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  cl_context context = command_queue->context;

  POCL_RETURN_ERROR_ON (
      (context->svm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND ((src_ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((dst_ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((region == NULL), CL_INVALID_VALUE);

  if (src_origin == NULL)
    src_origin = zero_origin;

  if (dst_origin == NULL)
    dst_origin = zero_origin;

  size_t region_bytes = region[0] * region[1] * region[2];
  POCL_RETURN_ERROR_ON ((region_bytes == 0), CL_INVALID_VALUE,
                        "All items in region must be >0\n");

  /*
    const char *s = (const char *)src_ptr;
    char *d = (char *)dst_ptr;
    if (((s <= d) && (s + size > d)) || ((d <= s) && (d + size > s)))
      POCL_RETURN_ERROR_ON (1, CL_MEM_COPY_OVERLAP, "overlapping copy \n");
  */
  size_t src_buf_size = 0;
  errcode = pocl_svm_check_pointer (context, src_ptr, 1, &src_buf_size);
  if (errcode != CL_SUCCESS)
    return errcode;
  /* even if we can't find the buffer, need to set up row_pitch+slice_pitch */
  if (src_buf_size == 0)
    src_buf_size = SIZE_MAX;
  errcode
      = pocl_buffer_boundcheck_3d (src_buf_size, src_origin, region,
                                   &src_row_pitch, &src_slice_pitch, "src_");
  if (errcode != CL_SUCCESS)
    return errcode;

  size_t dst_buf_size = 0;
  errcode = pocl_svm_check_pointer (context, dst_ptr, 1, &dst_buf_size);
  if (errcode != CL_SUCCESS)
    return errcode;
  if (dst_buf_size == 0)
    dst_buf_size = SIZE_MAX;
  errcode
      = pocl_buffer_boundcheck_3d (dst_buf_size, dst_origin, region,
                                   &dst_row_pitch, &dst_slice_pitch, "dst_");
  if (errcode != CL_SUCCESS)
    return errcode;

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_SVM_MEMCPY_RECT_POCL, event,
        num_items_in_wait_list, event_wait_list, NULL, 0);
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_SVM_MEMCPY_RECT_POCL,
        num_items_in_wait_list, sync_point_wait_list, NULL, 0);
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;

  c->command.svm_memcpy_rect.src = src_ptr;
  c->command.svm_memcpy_rect.dst = dst_ptr;
  memcpy (c->command.svm_memcpy_rect.region, region, 3 * sizeof (size_t));
  memcpy (c->command.svm_memcpy_rect.src_origin, src_origin,
          3 * sizeof (size_t));
  memcpy (c->command.svm_memcpy_rect.dst_origin, dst_origin,
          3 * sizeof (size_t));
  c->command.svm_memcpy_rect.dst_row_pitch = dst_row_pitch;
  c->command.svm_memcpy_rect.dst_slice_pitch = dst_slice_pitch;
  c->command.svm_memcpy_rect.src_row_pitch = src_row_pitch;
  c->command.svm_memcpy_rect.src_slice_pitch = src_slice_pitch;

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueSVMMemcpyRectPOCL) (cl_command_queue command_queue,
                                     cl_bool blocking,
                                     void *dst_ptr,
                                     const void *src_ptr,
                                     const size_t *dst_origin,
                                     const size_t *src_origin,
                                     const size_t *region,
                                     size_t dst_row_pitch,
                                     size_t dst_slice_pitch,
                                     size_t src_row_pitch,
                                     size_t src_slice_pitch,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list,
                                     cl_event *event)
    CL_API_SUFFIX__VERSION_2_0
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  errcode = pocl_svm_memcpy_rect_common (
      NULL, command_queue, dst_ptr, src_ptr, dst_origin, src_origin, region,
      dst_row_pitch, dst_slice_pitch, src_row_pitch, src_slice_pitch,
      num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  if (blocking)
    POname (clFinish) (command_queue);

  return CL_SUCCESS;
}
POsym (clEnqueueSVMMemcpyRectPOCL)
