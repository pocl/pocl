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

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseCommandQueue)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  int new_refcount;
  cl_context context = command_queue->context;
  cl_device_id device = command_queue->device;

  POname(clFlush)(command_queue);
  POCL_RELEASE_OBJECT(command_queue, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Command Queue %p  %d\n", command_queue, new_refcount);

  if (new_refcount == 0)
    {
      assert (command_queue->command_count == 0);
      POCL_MSG_PRINT_REFCOUNTS ("Free Command Queue %p\n", command_queue);
      if (command_queue->device->ops->free_queue)
        command_queue->device->ops->free_queue (device, command_queue);
      POCL_DESTROY_OBJECT (command_queue);
      POCL_MEM_FREE(command_queue);

      POname(clReleaseContext) (context);
    }
  return CL_SUCCESS;
}
POsym(clReleaseCommandQueue)
