/* OpenCL runtime library: clEnqueueSVMMemcpy()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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
POname(clEnqueueSVMMemcpy) (cl_command_queue command_queue,
                    cl_bool blocking_copy,
                    void *dst_ptr,
                    const void *src_ptr,
                    size_t size,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_2_0
{
  unsigned i;
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_ON((command_queue->context->svm_allocdev == NULL),
      CL_INVALID_CONTEXT, "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND((src_ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND((dst_ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND((size == 0), CL_INVALID_VALUE);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  for(i=0; i<num_events_in_wait_list; i++)
    POCL_RETURN_ERROR_COND((event_wait_list[i] == NULL), CL_INVALID_EVENT_WAIT_LIST);

  const char *s = (const char *)src_ptr;
  char *d = (char *)dst_ptr;
  if (((s <= d) && (s + size > d)) || ((d <= s) && (d + size > s)))
    POCL_RETURN_ERROR_ON (1, CL_MEM_COPY_OVERLAP, "overlapping copy \n");

  _cl_command_node *cmd = NULL;
  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_SVM_MEMCPY,
                                 event, num_events_in_wait_list,
                                 event_wait_list, 0, NULL, NULL);

  if (errcode != CL_SUCCESS)
    {
      POCL_MEM_FREE (cmd);
      return errcode;
    }

  cmd->command.svm_memcpy.src = src_ptr;
  cmd->command.svm_memcpy.dst = dst_ptr;
  cmd->command.svm_memcpy.size = size;

  pocl_command_enqueue(command_queue, cmd);

  if (blocking_copy == CL_TRUE)
    return POname (clFinish) (command_queue);
  else
    return CL_SUCCESS;
}
POsym(clEnqueueSVMMemcpy)

