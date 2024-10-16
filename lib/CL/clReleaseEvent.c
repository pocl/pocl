/* OpenCL runtime library: clReleaseEvent()

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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
#include "pocl_mem_management.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseEvent)(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (event)), CL_INVALID_EVENT);

  POCL_LOCK_OBJ (event);
  POCL_RELEASE_OBJECT_UNLOCKED (event, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Event %" PRIu64 " (%p), Refcount: %d\n",
                            event->id, event, new_refcount);

  if (new_refcount == 0)
    {
      POCL_UNLOCK_OBJ (event);
      VG_REFC_ZERO (event);

      assert (event->callback_list == NULL);
      if (event->command_type == CL_COMMAND_USER)
        {
          POCL_ATOMIC_DEC (uevent_c);
          pocl_user_event_data *p = (pocl_user_event_data *)event->data;
          POCL_DESTROY_COND (p->wakeup_cond);
          POCL_MEM_FREE (p);
        }
      else
        POCL_ATOMIC_DEC (event_c);

      POCL_MSG_PRINT_REFCOUNTS ("Free Event %" PRIu64 " (%p)\n", event->id,
                                event);
      if (event->command_type != CL_COMMAND_USER
          && event->queue->device->ops->free_event_data
          && (*(event->queue->device->available) == CL_TRUE))
        event->queue->device->ops->free_event_data(event);

      if (event->queue)
        POname(clReleaseCommandQueue) (event->queue);
      else
        POname(clReleaseContext) (event->context);

      POCL_DESTROY_OBJECT (event);
      pocl_mem_manager_free_event (event);
    }
  else
    {
      VG_REFC_NONZERO (event);
      POCL_UNLOCK_OBJ (event);
    }

  return CL_SUCCESS;
}
POsym(clReleaseEvent)
