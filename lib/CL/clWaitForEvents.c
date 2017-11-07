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

CL_API_ENTRY cl_int CL_API_CALL
POname(clWaitForEvents)(cl_uint              num_events ,
                  const cl_event *     event_list ) CL_API_SUFFIX__VERSION_1_0
{
  unsigned event_i;
  cl_device_id dev;
  cl_int ret = CL_SUCCESS;
  POCL_RETURN_ERROR_COND((num_events == 0 || event_list == NULL), CL_INVALID_VALUE);

  for (event_i = 0; event_i < num_events; ++event_i)
    {
      POCL_RETURN_ERROR_COND((event_list[event_i] == NULL), CL_INVALID_EVENT);
      if (event_i > 0)
        {
          POCL_RETURN_ERROR_COND((event_list[event_i]->context != event_list[event_i - 1]->context), CL_INVALID_CONTEXT);
        }
    }

  // dummy implementation, waits until *all* events have completed.
  for (event_i = 0; event_i < num_events; ++event_i)
    {
      /* lets handle user events later */
      if (event_list[event_i]->command_type == CL_COMMAND_USER)
        continue;
      dev = event_list[event_i]->queue->device;
      if (dev->ops->wait_event)  
        dev->ops->wait_event(dev, event_list[event_i]);
      else
        POname(clFinish)(event_list[event_i]->queue);
      if (event_list[event_i]->status < 0)
        ret = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
    }
  /* brute force wait for user events */
  struct timespec time_to_wait = { 0, 0 };
  for (event_i = 0; event_i < num_events; ++event_i)
    if (event_list[event_i]->command_type == CL_COMMAND_USER)
      {
        while (event_list[event_i]->status > CL_COMPLETE)
          {
            pocl_user_event_data *p = event_list[event_i]->data;
            POCL_LOCK (p->lock);
            time_to_wait.tv_sec = time (NULL) + 1;
            pthread_cond_timedwait (&p->wakeup_cond, &p->lock, &time_to_wait);
            POCL_UNLOCK (p->lock);
          }
        if (event_list[event_i]->status < 0)
          ret = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
      }

  return ret;
}
POsym(clWaitForEvents)
