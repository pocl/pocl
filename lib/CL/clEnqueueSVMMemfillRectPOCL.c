/* OpenCL runtime library: clEnqueueSVMMemfillRectPOCL()

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
pocl_svm_memfill_rect_common (cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              void *svm_ptr,
                              const size_t *origin,
                              const size_t *region,
                              size_t row_pitch,
                              size_t slice_pitch,
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
  const size_t zero_origin[3] = { 0, 0, 0 };

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  cl_context context = command_queue->context;

  POCL_RETURN_ERROR_ON (
      (context->svm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND ((svm_ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((pattern_size == 0), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((pattern_size > 128), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((region == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((pattern == NULL), CL_INVALID_VALUE);

  if (origin == NULL)
    origin = zero_origin;

  POCL_RETURN_ERROR_ON (
      (__builtin_popcount (pattern_size) > 1), CL_INVALID_VALUE,
      "pattern_size (%zu) must be a power-of-2 value\n", pattern_size);

  POCL_RETURN_ERROR_ON (((intptr_t)svm_ptr % pattern_size > 0),
                        CL_INVALID_VALUE,
                        "svm_ptr must be aligned to pattern_size\n");

  size_t region_bytes = region[0] * region[1] * region[2];
  POCL_RETURN_ERROR_ON ((region_bytes == 0), CL_INVALID_VALUE,
                        "All items in region must be >0\n");

  POCL_RETURN_ERROR_ON ((region[0] % pattern_size > 0), CL_INVALID_VALUE,
                        "region[0] must be a multiple of pattern_size\n");

  size_t buf_size = 0;
  errcode = pocl_svm_check_pointer (context, svm_ptr, 1, &buf_size);
  if (errcode != CL_SUCCESS)
    return errcode;
  /* even if we can't find the buffer, need to set up row_pitch+slice_pitch */
  if (buf_size == 0)
    buf_size = SIZE_MAX;

  errcode = pocl_buffer_boundcheck_3d (buf_size, origin, region, &row_pitch,
                                       &slice_pitch, "svm_");
  if (errcode != CL_SUCCESS)
    return errcode;

  void *cmd_pattern = pocl_aligned_malloc (pattern_size, pattern_size);
  POCL_RETURN_ERROR_COND ((cmd_pattern == NULL), CL_OUT_OF_HOST_MEMORY);

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_SVM_MEMFILL_RECT_POCL, event,
        num_items_in_wait_list, event_wait_list, NULL);
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_SVM_MEMFILL_RECT_POCL,
        num_items_in_wait_list, sync_point_wait_list, NULL);
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;

  memcpy (cmd_pattern, pattern, pattern_size);
  c->command.svm_fill_rect.svm_ptr = svm_ptr;
  memcpy (c->command.svm_fill_rect.region, region, 3 * sizeof (size_t));
  memcpy (c->command.svm_fill_rect.origin, origin, 3 * sizeof (size_t));
  c->command.svm_fill_rect.row_pitch = row_pitch;
  c->command.svm_fill_rect.slice_pitch = slice_pitch;
  c->command.svm_fill_rect.pattern = cmd_pattern;
  c->command.svm_fill_rect.pattern_size = pattern_size;

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueSVMMemFillRectPOCL) (cl_command_queue command_queue,
                                      void *svm_ptr,
                                      const size_t *origin,
                                      const size_t *region,
                                      size_t row_pitch,
                                      size_t slice_pitch,
                                      const void *pattern,
                                      size_t pattern_size,
                                      size_t size,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event)
    CL_API_SUFFIX__VERSION_2_0
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  errcode = pocl_svm_memfill_rect_common (
      NULL, command_queue, svm_ptr, origin, region, row_pitch, slice_pitch,
      pattern, pattern_size, num_events_in_wait_list, event_wait_list, event,
      NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym (clEnqueueSVMMemFillRectPOCL)
