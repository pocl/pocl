/* OpenCL runtime library: clCreateKernel()

   Copyright (c) 2011-2024 PoCL developers

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

#include "common.h"
#include "pocl_cl.h"
#include "pocl_timing.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetUserEventStatus)(cl_event event ,
                             cl_int   execution_status)
CL_API_SUFFIX__VERSION_1_1
{
  int errcode = CL_SUCCESS;
  /* Must be a valid user event */
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (event)), CL_INVALID_EVENT);
  /* Can only be set to CL_COMPLETE (0) or negative values */
  POCL_RETURN_ERROR_COND ((execution_status > CL_COMPLETE), CL_INVALID_VALUE);

  POCL_LOCK_OBJ (event);

  POCL_GOTO_ERROR_COND ((event->command_type != CL_COMMAND_USER),
                        CL_INVALID_EVENT);
  /* Can only be done once */
  POCL_GOTO_ERROR_COND ((event->status <= CL_COMPLETE), CL_INVALID_OPERATION);

  event->status = execution_status;
  POCL_UNLOCK_OBJ (event);

  if (execution_status <= CL_COMPLETE)
    {
      POCL_MSG_PRINT_EVENTS ("User event %" PRIu64 " completed with status: %i\n",
                             event->id, execution_status);
      pocl_broadcast (event);
    }

  POCL_LOCK_OBJ (event);
  pocl_event_updated (event, execution_status);
  pocl_user_event_data *p = (pocl_user_event_data *)event->data;
  if (execution_status <= CL_COMPLETE)
    POCL_BROADCAST_COND (p->wakeup_cond);

ERROR:
  POCL_UNLOCK_OBJ (event);
  return errcode;
}
POsym(clSetUserEventStatus)
