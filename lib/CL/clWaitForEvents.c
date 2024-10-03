/* OpenCL runtime library: clWaitForEvents()

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
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clWaitForEvents)(cl_uint              num_events ,
                  const cl_event *     event_list ) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i;
  cl_device_id dev;
  cl_int ret = CL_SUCCESS;

  POCL_RETURN_ERROR_COND((num_events == 0 || event_list == NULL), CL_INVALID_VALUE);

  for (i = 0; i < num_events; ++i)
    {
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (event_list[i])),
                              CL_INVALID_EVENT);
      if (i > 0)
        {
          POCL_RETURN_ERROR_COND (
              (event_list[i]->context != event_list[i - 1]->context),
              CL_INVALID_CONTEXT);
        }
    }

  // dummy implementation, waits until *all* events have completed.
  for (i = 0; i < num_events; ++i)
    {
      /* lets handle user events later */
      if (event_list[i]->command_type == CL_COMMAND_USER)
        continue;
      dev = event_list[i]->queue->device;
      POCL_RETURN_ERROR_COND ((*(dev->available) == CL_FALSE),
                              CL_DEVICE_NOT_AVAILABLE);
      /* this is necessary, man clFlush says:
       * Any blocking commands .. perform an implicit flush of the cmd queue.
       * To use event objects that refer to commands enqueued in a cmd queue
       * as event objects to wait on by commands enqueued in a different
       * command-queue, the application must call a clFlush or any blocking
       * commands that perform an implicit flush */
      POname(clFlush) (event_list[i]->queue);
      if (dev->ops->wait_event)
        dev->ops->wait_event (dev, event_list[i]);
      else
        POname(clFinish) (event_list[i]->queue);
      if (event_list[i]->status < 0)
        ret = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
    }

  if (ret != CL_SUCCESS)
    return ret;

  /* wait for user events */
  for (i = 0; i < num_events; ++i)
    {
      cl_event e = event_list[i];
      POCL_LOCK_OBJ (e);
      pocl_user_event_data *p = (pocl_user_event_data *)e->data;
      if (e->command_type == CL_COMMAND_USER)
        {
          while (e->status > CL_COMPLETE)
            {
              POCL_WAIT_COND (p->wakeup_cond, e->pocl_lock);
            }
          if (e->status < 0)
            ret = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
        }
      POCL_UNLOCK_OBJ (e);
    }

  return ret;
}
POsym(clWaitForEvents)
