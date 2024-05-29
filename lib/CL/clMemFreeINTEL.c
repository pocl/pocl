/* OpenCL runtime library: clMemFreeINTEL() / clMemBlockingFreeINTEL()

   Copyright (c) 2023 Michal Babej / Intel Finland Oy
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_debug.h"
#include "pocl_util.h"
#include "utlist.h"

extern unsigned long usm_buffer_c;

/* get & retain all last-events of all command queues of the context */
static int
pocl_get_last_events (cl_context context, cl_event **last_events,
                      unsigned *last_event_count)
{
  /* store last events for all queues in the context */
  cl_command_queue temp_cq = NULL;
  cl_event temp_ev = NULL;
  unsigned le_count = 0, cq_count = 0;
  DL_FOREACH (context->command_queues, temp_cq) { ++cq_count; }
  cq_count += context->num_devices;
  cl_event *levents = malloc (cq_count * sizeof (cl_event));
  POCL_RETURN_ERROR_COND ((levents == NULL), CL_OUT_OF_HOST_MEMORY);

  DL_FOREACH (context->command_queues, temp_cq)
  {
    POCL_LOCK_OBJ (temp_cq);
    temp_ev = temp_cq->last_event.event;
    if (temp_ev)
      {
        levents[le_count++] = temp_ev;
        POname (clRetainEvent) (temp_ev);
      }
    POCL_UNLOCK_OBJ (temp_cq);
  }

  for (unsigned i = 0; i < context->num_devices; ++i)
    {
      cl_device_id dev = context->devices[i];
      if (context->default_queues && context->default_queues[i])
        {
          POCL_LOCK_OBJ (context->default_queues[i]);
          temp_ev = context->default_queues[i]->last_event.event;
          if (temp_ev)
            {
              levents[le_count++] = temp_ev;
              POname (clRetainEvent) (temp_ev);
            }
          POCL_UNLOCK_OBJ (context->default_queues[i]);
        }
    }
  *last_events = levents;
  *last_event_count = le_count;
  return CL_SUCCESS;
}

static int
pocl_mem_free_intel (cl_context context, void *usm_pointer, cl_bool blocking)
{
  POCL_RETURN_ERROR_COND (!IS_CL_OBJECT_VALID (context), CL_INVALID_CONTEXT);

  POCL_RETURN_ERROR_ON (
      (context->usm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is USM-capable\n");

  if (usm_pointer == NULL)
    {
      POCL_MSG_WARN ("NULL pointer passed\n");
      return CL_SUCCESS;
    }

  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *tmp = NULL, *item = NULL;
  DL_FOREACH_SAFE (context->raw_ptrs, item, tmp)
  {
    if (item->vm_ptr == usm_pointer)
      {
        DL_DELETE (context->raw_ptrs, item);
        break;
      }
  }
  POCL_UNLOCK_OBJ (context);
  POCL_RETURN_ERROR_ON (
      (item == NULL), CL_INVALID_VALUE,
      "Can't find pointer in list of allocated USM pointers");

  if (blocking == CL_FALSE)
    {
      context->usm_allocdev->ops->usm_free (context->usm_allocdev,
                                            usm_pointer);
    }
  else
    {
      /* if the device implements blocking free callback */
      if (context->usm_allocdev->ops->usm_free_blocking)
        context->usm_allocdev->ops->usm_free_blocking (context->usm_allocdev,
                                                       usm_pointer);
      else
        {
          /* otherwise wait for all queues in the context */
          cl_event *last_events = NULL;
          unsigned last_event_count = 0;
          POCL_LOCK_OBJ (context);
          int err = pocl_get_last_events (context, &last_events,
                                          &last_event_count);
          POCL_UNLOCK_OBJ (context);
          if (err != CL_SUCCESS)
            return err;
          if (last_event_count > 0)
            {
              POname (clWaitForEvents) (last_event_count, last_events);
              for (unsigned i = 0; i < last_event_count; ++i)
                {
                  POname (clReleaseEvent) (last_events[i]);
                }
            }
          context->usm_allocdev->ops->usm_free (context->usm_allocdev,
                                                usm_pointer);
        }
    }

  POCL_MEM_FREE (item);
  POname (clReleaseContext) (context);

  POCL_ATOMIC_DEC (usm_buffer_c);

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clMemFreeINTEL) (cl_context context,
                         void *usm_pointer) CL_API_SUFFIX__VERSION_2_0
{
  return pocl_mem_free_intel (context, usm_pointer, CL_FALSE);
}
POsym (clMemFreeINTEL)

    CL_API_ENTRY cl_int CL_API_CALL
    POname (clMemBlockingFreeINTEL) (cl_context context, void *usm_pointer)
        CL_API_SUFFIX__VERSION_2_0
{
  return pocl_mem_free_intel (context, usm_pointer, CL_TRUE);
}
POsym (clMemBlockingFreeINTEL)
