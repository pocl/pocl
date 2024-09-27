/* OpenCL runtime library: clFinish()

   Copyright (c) 2011 Erik Schnetter

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
#include "pocl_debug.h"
#include "pocl_image_util.h"
#include "utlist.h"
#include "pocl_shared.h"
#include "common.h"
#include "pocl_mem_management.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clFinish)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
#ifdef POCL_DEBUG_MESSAGES
  if (pocl_get_bool_option ("POCL_DUMP_TASK_GRAPHS", 0) == 1)
    {
      pocl_dump_dot_task_graph (command_queue->context, "pocl-task-graph.dot");
      pocl_dump_dot_task_graph_signal ();
    }
#endif
  /* Flush all pending commands */
  int err = POname (clFlush) (command_queue);
  if (err != CL_SUCCESS)
    return err;

  POCL_LOCK_OBJ (command_queue);
  ++command_queue->notification_waiting_threads;
  POCL_RETAIN_OBJECT_UNLOCKED (command_queue);
  POCL_UNLOCK_OBJ (command_queue);

  command_queue->device->ops->join(command_queue->device, command_queue);

  POCL_LOCK_OBJ (command_queue);
  --command_queue->notification_waiting_threads;
  POCL_UNLOCK_OBJ (command_queue);
  POname (clReleaseCommandQueue) (command_queue);

  return CL_SUCCESS;
}
POsym(clFinish)
