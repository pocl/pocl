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
  if (image == NULL)
    return CL_INVALID_MEM_OBJECT;

  if (event != NULL)
    {
      *event = (cl_event)malloc(sizeof(struct _cl_event));
      if (*event == NULL)
        return CL_OUT_OF_HOST_MEMORY; 
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      (*event)->command_type = CL_COMMAND_READ_IMAGE;

      POname(clRetainCommandQueue) (command_queue);
      POCL_UPDATE_EVENT_QUEUED;
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
    }
  
  if (blocking_read)
    {
      status = pocl_read_image(image, command_queue->device, origin, region,
                               host_row_pitch, host_slice_pitch, ptr);
      POCL_UPDATE_EVENT_COMPLETE;
      return status;
    }
  else /* non blocking */
    {
      POCL_ABORT_UNIMPLEMENTED();
    }
}
POsym(clEnqueueReadImage)
