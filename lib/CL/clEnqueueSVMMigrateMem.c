/* OpenCL runtime library: clEnqueueSVMMigrateMem()

   Copyright (c) 2022 Michal Babej / Tampere University of Technology

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

#include "pocl_cl.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueSVMMigrateMem) (cl_command_queue command_queue,
                                 cl_uint num_svm_pointers,
                                 const void **svm_pointers,
                                 const size_t *sizes,
                                 cl_mem_migration_flags flags,
                                 cl_uint num_events_in_wait_list,
                                 const cl_event *event_wait_list,
                                 cl_event *event) CL_API_SUFFIX__VERSION_2_1
{
  unsigned i;
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_ON (
      (command_queue->context->svm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND ((svm_pointers == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((num_svm_pointers == 0), CL_INVALID_VALUE);

  cl_mem_migration_flags not_valid_flags = ~(
      CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
  POCL_RETURN_ERROR_ON ((flags & not_valid_flags), CL_INVALID_VALUE,
                        "invalid flags given\n");

  for (i = 0; i < num_svm_pointers; ++i)
    {
      POCL_RETURN_ERROR_COND ((svm_pointers[i] == NULL), CL_INVALID_VALUE);
    }

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *cmd = NULL;
  errcode = pocl_create_command (
      &cmd, command_queue, CL_COMMAND_SVM_MIGRATE_MEM, event,
      num_events_in_wait_list, event_wait_list, 0, NULL, NULL);

  if (errcode != CL_SUCCESS)
    {
      POCL_MEM_FREE (cmd);
      return errcode;
    }

  if (sizes)
    {
      size_t *s = malloc (num_svm_pointers * sizeof (size_t));
      memcpy (s, sizes, num_svm_pointers * sizeof (size_t));
      cmd->command.svm_migrate.sizes = s;
    }
  else
    cmd->command.svm_migrate.sizes = NULL;

  void **p = malloc (num_svm_pointers * sizeof (void *));
  memcpy (p, svm_pointers, num_svm_pointers * sizeof (void *));
  cmd->command.svm_migrate.svm_pointers = p;
  cmd->command.svm_migrate.num_svm_pointers = num_svm_pointers;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym (clEnqueueSVMMigrateMem)
