/* OpenCL runtime library: clEnqueueMarker()

   Copyright (c) 2011 Pekka Jääskeläinen / Tampere Univ. of Tech.
   
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
#include "utlist.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueMarker)(cl_command_queue     command_queue,
                  cl_event *           event) 
CL_API_SUFFIX__VERSION_1_0
{
  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (event == NULL)
    return CL_INVALID_VALUE;

  *event = (cl_event)malloc(sizeof(struct _cl_event));
  if (*event == NULL)
    return CL_OUT_OF_HOST_MEMORY; 
  POCL_INIT_OBJECT(*event);
  (*event)->queue = command_queue;
  POname(clRetainCommandQueue) (command_queue);
  POCL_UPDATE_EVENT_QUEUED;

  _cl_command_node * cmd = malloc(sizeof(_cl_command_node));
  if (cmd == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  cmd->type = CL_COMMAND_MARKER;
  cmd->next = NULL;
  cmd->event = *event;
  LL_APPEND(command_queue->root, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueMarker) 
