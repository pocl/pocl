/* OpenCL runtime library: clEnqueueBarrierWithWaitList()

   Copyright (c) 2015 Ville Korhonen, TUT
   
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
#include "pocl_util.h"

CL_API_ENTRY 
cl_int CL_API_CALL
POname(clEnqueueBarrierWithWaitList)(cl_command_queue command_queue,
                                     cl_uint          num_events_in_wait_list,
                                     const cl_event   *event_wait_list,
                                     cl_event         *event) 
CL_API_SUFFIX__VERSION_1_2
{
  _cl_command_node *cmd;
  unsigned i;

  POCL_RETURN_ERROR_COND((event_wait_list == NULL && num_events_in_wait_list > 0),
    CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND((event_wait_list != NULL && num_events_in_wait_list == 0),
    CL_INVALID_EVENT_WAIT_LIST);

  for (i = 0; i < num_events_in_wait_list; i++)
    POCL_RETURN_ERROR_COND((event_wait_list[i] == NULL), CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND((command_queue->device == NULL), CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND((command_queue->context == NULL), CL_INVALID_COMMAND_QUEUE);

  /* Even if we do not need to create a full command, the runtime requires it */
  pocl_create_command (&cmd, command_queue, 
                       CL_COMMAND_BARRIER, NULL, 
                       num_events_in_wait_list, event_wait_list, 0, NULL);

  cmd->command.barrier.has_wait_list = num_events_in_wait_list;
  pocl_command_enqueue(command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueBarrierWithWaitList)
