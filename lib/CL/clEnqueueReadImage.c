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

  POCL_RETURN_ERROR_ON((image->buffer->flags & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)),
    CL_INVALID_OPERATION, "Image buffer has been created with CL_MEM_HOST_WRITE_ONLY "
    "or CL_MEM_HOST_NO_ACCESS\n");

  errcode = pocl_check_event_wait_list(command_queue, num_events_in_wait_list, event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_device_supports_image(image, command_queue);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;

  size_t tuned_origin[3] = {origin[0] * image->image_elem_size * image->image_channels, origin[1], 
                            origin[2]};
  size_t tuned_region[3] = {region[0] * image->image_elem_size * image->image_channels, region[1], 
                            region[2]};
  
  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_READ_IMAGE,
                                event, num_events_in_wait_list, 
                                event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    {
      POCL_MEM_FREE(cmd);
      return errcode;
    }

  cmd->command.read_image.device_ptr = 
    image->device_ptrs[command_queue->device->dev_id].mem_ptr;
  cmd->command.read_image.host_ptr = ptr;
  memcpy ((cmd->command.read_image.origin), tuned_origin, 3*sizeof (size_t));
  memcpy ((cmd->command.read_image.region), tuned_region, 3*sizeof (size_t));
  cmd->command.write_image.b_rowpitch = image->image_row_pitch;
  cmd->command.write_image.b_slicepitch = image->image_slice_pitch;
  cmd->command.write_image.h_rowpitch = (row_pitch ? row_pitch : tuned_region[0]);
  cmd->command.write_image.h_slicepitch = (slice_pitch ? slice_pitch : (tuned_region[0]*region[1]));
  cmd->command.read_image.buffer = image;

  POname(clRetainMemObject) (image);  
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_read)
    POname(clFinish) (command_queue);
  
  return errcode;
}
POsym(clEnqueueReadImage)
