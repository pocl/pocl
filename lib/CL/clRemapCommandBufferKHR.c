/* OpenCL runtime library: clRemapCommandBufferKHR()

   Copyright (c) 2022-2024 Jan Solanti / Tampere University

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
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
POname (clRemapCommandBufferKHR) (cl_command_buffer_khr command_buffer,
                                  cl_bool automatic,
                                  cl_uint num_queues,
                                  const cl_command_queue *queues,
                                  cl_uint num_handles,
                                  const cl_mutable_command_khr *handles,
                                  cl_mutable_command_khr *handles_ret,
                                  cl_int *errcode_ret)
  CL_API_SUFFIX__VERSION_1_2
{
  int errcode = 0;
  cl_command_buffer_khr new_cmdbuf = NULL;

  if ((errcode = pocl_cmdbuf_validate_queue_list (num_queues, queues))
      != CL_SUCCESS)
    {
      *errcode_ret = errcode;
      return NULL;
    }

  POCL_GOTO_ERROR_COND (
    (num_queues != command_buffer->num_queues && !automatic),
    CL_INVALID_VALUE);

  new_cmdbuf = POname (clCreateCommandBufferKHR) (
    num_queues, queues, command_buffer->properties, &errcode);
  if (errcode != CL_SUCCESS)
    {
      *errcode_ret = errcode;
      return NULL;
    }

  _cl_command_node *cmd;
  LL_FOREACH (command_buffer->cmds, cmd)
  {
    assert (cmd->buffered);
    /* TODO: be smarter about this */
    cl_uint new_queue_idx = cmd->queue_idx % new_cmdbuf->num_queues;
    cl_command_queue new_queue = queues[new_queue_idx];

    /* Simply re-record all commands. Syncpoints are plain integers assigned
     * in sequence so simply reusing the old wait list should be fine. */
    switch (cmd->type)
      {
      case CL_COMMAND_BARRIER:
        errcode = POname (clCommandBarrierWithWaitListKHR) (
          new_cmdbuf, new_queue, NULL,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_COPY_BUFFER:
        errcode = POname (clCommandCopyBufferKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.copy.src, cmd->command.copy.dst,
          cmd->command.copy.src_offset, cmd->command.copy.dst_offset,
          cmd->command.copy.size,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_COPY_BUFFER_RECT:
        errcode = POname (clCommandCopyBufferRectKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.copy_rect.src,
          cmd->command.copy_rect.dst, cmd->command.copy_rect.src_origin,
          cmd->command.copy_rect.dst_origin, cmd->command.copy_rect.region,
          cmd->command.copy_rect.src_row_pitch,
          cmd->command.copy_rect.src_slice_pitch,
          cmd->command.copy_rect.dst_row_pitch,
          cmd->command.copy_rect.dst_slice_pitch,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
        errcode = POname (clCommandCopyBufferToImageKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.write_image.src,
          cmd->command.write_image.dst, cmd->command.write_image.src_offset,
          cmd->command.write_image.origin, cmd->command.write_image.region,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
        errcode = POname (clCommandCopyImageToBufferKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.read_image.src,
          cmd->command.read_image.dst, cmd->command.read_image.origin,
          cmd->command.read_image.region, cmd->command.read_image.dst_offset,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_COPY_IMAGE:
        errcode = POname (clCommandCopyImageKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.copy_image.src,
          cmd->command.copy_image.dst, cmd->command.copy_image.src_origin,
          cmd->command.copy_image.dst_origin, cmd->command.copy_image.region,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;

      case CL_COMMAND_FILL_BUFFER:
        errcode = POname (clCommandFillBufferKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.memfill.dst,
          cmd->command.memfill.pattern, cmd->command.memfill.pattern_size,
          cmd->command.memfill.offset, cmd->command.memfill.size,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_FILL_IMAGE:
        errcode = POname (clCommandFillImageKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.fill_image.dst,
          cmd->command.fill_image.fill_pixel, cmd->command.fill_image.origin,
          cmd->command.fill_image.region,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;

      case CL_COMMAND_NDRANGE_KERNEL:
        {
          cl_uint work_dim = cmd->command.run.pc.work_dim;
          size_t *local_size = cmd->command.run.pc.local_size;
          size_t *groups = cmd->command.run.pc.num_groups;
          size_t global_size[3]
            = { local_size[0] * groups[0],
                work_dim > 1 ? (local_size[1] * groups[1]) : 0,
                work_dim > 2 ? (local_size[2] * groups[2]) : 0 };

          /* Re-record cmd using the original command's kernel arguments.
           *
           * TODO: pass along kernel command properties */
          errcode = pocl_record_ndrange_kernel (
            new_cmdbuf, new_queue, NULL, cmd->command.run.kernel,
            cmd->command.run.arguments, work_dim,
            cmd->command.run.pc.global_offset, global_size, local_size,
            cmd->sync.syncpoint.num_sync_points_in_wait_list,
            cmd->sync.syncpoint.sync_point_wait_list, NULL);
        }
        break;

      case CL_COMMAND_READ_BUFFER:
        errcode = POname (clCommandReadBufferPOCL) (
          new_cmdbuf, new_queue, cmd->command.read.src,
          cmd->command.read.offset, cmd->command.read.size,
          cmd->command.read.dst_host_ptr,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_READ_BUFFER_RECT:
        errcode = POname (clCommandReadBufferRectPOCL) (
          new_cmdbuf, new_queue, cmd->command.read_rect.src,
          cmd->command.read_rect.buffer_origin,
          cmd->command.read_rect.host_origin, cmd->command.read_rect.region,
          cmd->command.read_rect.buffer_row_pitch,
          cmd->command.read_rect.buffer_slice_pitch,
          cmd->command.read_rect.host_row_pitch,
          cmd->command.read_rect.host_slice_pitch,
          cmd->command.read_rect.dst_host_ptr,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_READ_IMAGE:
        errcode = POname (clCommandReadImagePOCL) (
          new_cmdbuf, new_queue, cmd->command.read_image.src,
          cmd->command.read_image.origin, cmd->command.read_image.region,
          cmd->command.read_image.dst_row_pitch,
          cmd->command.read_image.dst_slice_pitch,
          cmd->command.read_image.dst_host_ptr,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;

      case CL_COMMAND_SVM_MEMCPY:
        errcode = POname (clCommandSVMMemcpyKHR) (
          new_cmdbuf, new_queue, NULL, cmd->command.svm_memcpy.dst,
          cmd->command.svm_memcpy.src, cmd->command.svm_memcpy.size,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_SVM_MEMCPY_RECT_POCL:
        errcode = POname (clCommandSVMMemcpyRectPOCL) (
          new_cmdbuf, new_queue, cmd->command.svm_memcpy_rect.dst,
          cmd->command.svm_memcpy_rect.src,
          cmd->command.svm_memcpy_rect.dst_origin,
          cmd->command.svm_memcpy_rect.src_origin,
          cmd->command.svm_memcpy_rect.region,
          cmd->command.svm_memcpy_rect.dst_row_pitch,
          cmd->command.svm_memcpy_rect.dst_slice_pitch,
          cmd->command.svm_memcpy_rect.src_row_pitch,
          cmd->command.svm_memcpy_rect.src_slice_pitch,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_SVM_MEMFILL:
        errcode = POname (clCommandSVMMemfillPOCL) (
          new_cmdbuf, new_queue, cmd->command.svm_fill.svm_ptr,
          cmd->command.svm_fill.size, cmd->command.svm_fill.pattern,
          cmd->command.svm_fill.pattern_size,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_SVM_MEMFILL_RECT_POCL:
        errcode = POname (clCommandSVMMemfillRectPOCL) (
          new_cmdbuf, new_queue, cmd->command.svm_fill_rect.svm_ptr,
          cmd->command.svm_fill_rect.origin, cmd->command.svm_fill_rect.region,
          cmd->command.svm_fill_rect.row_pitch,
          cmd->command.svm_fill_rect.slice_pitch,
          cmd->command.svm_fill_rect.pattern,
          cmd->command.svm_fill_rect.pattern_size,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;

      case CL_COMMAND_WRITE_BUFFER:
        errcode = POname (clCommandWriteBufferPOCL) (
          new_cmdbuf, new_queue, cmd->command.write.dst,
          cmd->command.write.offset, cmd->command.write.size,
          cmd->command.write.src_host_ptr,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_WRITE_BUFFER_RECT:
        errcode = POname (clCommandWriteBufferRectPOCL) (
          new_cmdbuf, new_queue, cmd->command.write_rect.dst,
          cmd->command.write_rect.buffer_origin,
          cmd->command.write_rect.host_origin, cmd->command.write_rect.region,
          cmd->command.write_rect.buffer_row_pitch,
          cmd->command.write_rect.buffer_slice_pitch,
          cmd->command.write_rect.host_row_pitch,
          cmd->command.write_rect.host_slice_pitch,
          cmd->command.write_rect.src_host_ptr,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;
      case CL_COMMAND_WRITE_IMAGE:
        errcode = POname (clCommandWriteImagePOCL) (
          new_cmdbuf, new_queue, cmd->command.write_image.dst,
          cmd->command.write_image.origin, cmd->command.write_image.region,
          cmd->command.write_image.src_row_pitch,
          cmd->command.write_image.src_slice_pitch,
          cmd->command.write_image.src_host_ptr,
          cmd->sync.syncpoint.num_sync_points_in_wait_list,
          cmd->sync.syncpoint.sync_point_wait_list, NULL, NULL);
        break;

      default:
        assert (0 && "Unhandled command in command buffer");
        errcode = CL_INVALID_OPERATION;
        break;
      }
    if (errcode != CL_SUCCESS)
      goto ERROR;
  }

  if (command_buffer->state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR
      || command_buffer->state == CL_COMMAND_BUFFER_STATE_PENDING_KHR)
    {
      errcode = POname (clFinalizeCommandBufferKHR) (new_cmdbuf);
      if (errcode != CL_SUCCESS)
        goto ERROR;
    }

  if (errcode_ret != NULL)
    *errcode_ret = errcode;
  return new_cmdbuf;

ERROR:
  POname (clReleaseCommandBufferKHR) (new_cmdbuf);
  if (errcode_ret != NULL)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clRemapCommandBufferKHR)
