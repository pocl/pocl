/* OpenCL runtime library: clEnqueueSVMMap()

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
POname(clEnqueueSVMMap) (cl_command_queue command_queue,
                 cl_bool blocking_map,
                 cl_map_flags map_flags,
                 void *svm_ptr,
                 size_t size,
                 cl_uint num_events_in_wait_list,
                 const cl_event *event_wait_list,
                 cl_event *event) CL_API_SUFFIX__VERSION_2_0
{
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  cl_context context = command_queue->context;

  POCL_RETURN_ERROR_ON (
      (context->svm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND ((size == 0), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((svm_ptr == NULL), CL_INVALID_VALUE);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_svm_check_pointer (context, svm_ptr, size, NULL);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (DEVICE_MMAP_IS_NOP(command_queue->device)
      && (num_events_in_wait_list == 0)
      && (event == NULL))
    {
      if (blocking_map == CL_TRUE)
        return POname(clFinish)(command_queue);
      else
        return CL_SUCCESS;
    }

  _cl_command_node *cmd = NULL;

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_SVM_MAP,
                                 event, num_events_in_wait_list,
                                 event_wait_list, 0, NULL, NULL);

  if (errcode != CL_SUCCESS)
    {
      POCL_MEM_FREE(cmd);
      return errcode;
    }

  cmd->command.svm_map.svm_ptr = svm_ptr;
  cmd->command.svm_map.size = size;
  cmd->command.svm_map.flags = map_flags;

  pocl_command_enqueue(command_queue, cmd);

  if (blocking_map == CL_TRUE)
    return POname(clFinish)(command_queue);
  else
    return CL_SUCCESS;

}
POsym(clEnqueueSVMMap)

