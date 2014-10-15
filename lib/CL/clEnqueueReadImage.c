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
                           size_t               host_row_pitch,
                           size_t               host_slice_pitch, 
                           void *               ptr,
                           cl_uint              num_events_in_wait_list,
                           const cl_event *     event_wait_list,
                           cl_event *           event) 
CL_API_SUFFIX__VERSION_1_0 
{
  cl_int status;
  _cl_command_node *cmd;

  if (image == NULL)
    return CL_INVALID_MEM_OBJECT;

  status = pocl_check_image_origin_region (image, origin, region);
  if (status != CL_SUCCESS)
    return status;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (command_queue->context != image->context)
    return CL_INVALID_CONTEXT;

  size_t tuned_origin[3] = {origin[0] * image->image_elem_size * image->image_channels, origin[1], 
                            origin[2]};
  size_t tuned_region[3] = {region[0] * image->image_elem_size * image->image_channels, region[1], 
                            region[2]};
  
  status = pocl_create_command (&cmd, command_queue, CL_COMMAND_READ_IMAGE, 
                                event, num_events_in_wait_list, 
                                event_wait_list);
  if (status != CL_SUCCESS)
    {
      if (event)
        POCL_MEM_FREE(*event);
    }

  cmd->command.rw_image.device_ptr = 
    image->device_ptrs[command_queue->device->dev_id].mem_ptr;
  cmd->command.rw_image.host_ptr = ptr;
  memcpy ((cmd->command.rw_image.origin), tuned_origin, 3*sizeof (size_t));
  memcpy ((cmd->command.rw_image.region), tuned_region, 3*sizeof (size_t));
  cmd->command.rw_image.rowpitch = image->image_row_pitch;
  cmd->command.rw_image.slicepitch = image->image_slice_pitch;
  cmd->command.rw_image.buffer = image;
  pocl_command_enqueue (command_queue, cmd);
  POname(clRetainMemObject) (image);  

  if (blocking_read)
    POname(clFinish) (command_queue);
  
  return status;
}
POsym(clEnqueueReadImage)
