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
  /* Flush all pending commands */
  int err = POname (clFlush) (command_queue);
  if (err != CL_SUCCESS)
    return err;

  command_queue->device->ops->join(command_queue->device, command_queue);

  return CL_SUCCESS;
}
POsym(clFinish)
