/* OpenCL runtime library: clEnqueueReadImage()

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

#include "pocl_cl.h"
#include "assert.h"
#include "pocl_image_util.h"
#include "pocl_util.h"
#include "utlist.h"
#include <string.h>

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueReadImage)(cl_command_queue     command_queue,
                           cl_mem               image,
                           cl_bool              blocking_read, 
                           const size_t *       origin, /* [3] */
                           const size_t *       region, /* [3] */
                           size_t               row_pitch,
                           size_t               slice_pitch,
                           void *               ptr,
                           cl_uint              num_events_in_wait_list,
                           const cl_event *     event_wait_list,
                           cl_event *           event) 
CL_API_SUFFIX__VERSION_1_0 
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND((ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((command_queue->context != image->context),
    CL_INVALID_CONTEXT, "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON (
      (!command_queue->device->image_support), CL_INVALID_OPERATION,
      "Device %s does not support images\n", command_queue->device->long_name);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_ON (
      (image->flags & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)),
      CL_INVALID_OPERATION,
      "image has been created with CL_MEM_HOST_WRITE_ONLY "
      "or CL_MEM_HOST_NO_ACCESS\n");

  if (image->buffer)
    POCL_RETURN_ERROR_ON (
        (image->buffer->flags
         & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)),
        CL_INVALID_OPERATION,
        "1D Image buffer has been created with CL_MEM_HOST_WRITE_ONLY "
        "or CL_MEM_HOST_NO_ACCESS\n");

  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;
  
  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_READ_IMAGE,
                                event, num_events_in_wait_list, 
                                event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    {
      POCL_MEM_FREE(cmd);
      return errcode;
    }

  HANDLE_IMAGE1D_BUFFER (image);

  cl_device_id dev = command_queue->device;
  cmd->command.read_image.src_mem_id = &image->device_ptrs[dev->dev_id];
  cmd->command.read_image.dst_host_ptr = ptr;
  cmd->command.read_image.dst_mem_id = NULL;

  cmd->command.read_image.origin[0] = origin[0];
  cmd->command.read_image.origin[1] = origin[1];
  cmd->command.read_image.origin[2] = origin[2];
  cmd->command.read_image.region[0] = region[0];
  cmd->command.read_image.region[1] = region[1];
  cmd->command.read_image.region[2] = region[2];

  cmd->command.read_image.dst_row_pitch = row_pitch;
  cmd->command.read_image.dst_slice_pitch = slice_pitch;
  cmd->command.read_image.dst_offset = 0;

  POname(clRetainMemObject) (image);  
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_read)
    POname(clFinish) (command_queue);
  
  return errcode;
}
POsym(clEnqueueReadImage)
