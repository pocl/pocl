/* OpenCL runtime library: clReleaseCommandQueue()

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and Pekka Jääskeläinen
   
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

extern unsigned long queue_c;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseCommandQueue)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  int new_refcount;
  cl_context context = command_queue->context;
  cl_device_id device = command_queue->device;

  POname(clFlush)(command_queue);
  POCL_LOCK_OBJ (command_queue);
  POCL_RELEASE_OBJECT_UNLOCKED (command_queue, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Command Queue %" PRId64
                            " (%p), Refcount: %d\n",
                            command_queue->id, command_queue, new_refcount);

  if (new_refcount == 0)
    {
      POCL_UNLOCK_OBJ (command_queue);
      VG_REFC_ZERO (command_queue);

      TP_FREE_QUEUE (context->id, command_queue->id);

      POCL_ATOMIC_DEC (queue_c);

      /* hidden queues don't retain the context. */
      if ((command_queue->properties & CL_QUEUE_HIDDEN) == 0)
        {
          POCL_LOCK_OBJ (context);
          DL_DELETE (context->command_queues, command_queue);
          POCL_UNLOCK_OBJ (context);

          POname (clReleaseContext) (context);
        }

      assert (command_queue->command_count == 0);
      POCL_MSG_PRINT_REFCOUNTS ("Free Command Queue %" PRId64 " (%p)\n",
                                command_queue->id, command_queue);
      if (command_queue->device->ops->free_queue
          && (*(command_queue->device->available) == CL_TRUE))
        command_queue->device->ops->free_queue (device, command_queue);
      POCL_DESTROY_OBJECT (command_queue);
      POCL_MEM_FREE(command_queue);
    }
  else
    {
      VG_REFC_NONZERO (command_queue);
      POCL_UNLOCK_OBJ (command_queue);
    }
  return CL_SUCCESS;
}
POsym(clReleaseCommandQueue)
