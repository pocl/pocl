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

cl_int
pocl_svm_migrate_mem_common (cl_command_type command_type,
                             cl_command_queue command_queue,
                             cl_uint num_svm_pointers,
                             const void **svm_pointers, const size_t *sizes,
                             cl_mem_migration_flags flags,
                             cl_uint num_events_in_wait_list,
                             const cl_event *event_wait_list, cl_event *event)
{
  unsigned i;
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  cl_context context = command_queue->context;

  POCL_RETURN_ERROR_ON (
      (context->svm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND ((svm_pointers == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((num_svm_pointers == 0), CL_INVALID_VALUE);

  cl_mem_migration_flags not_valid_flags = ~(
      CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
  POCL_RETURN_ERROR_ON ((flags & not_valid_flags), CL_INVALID_VALUE,
                        "invalid flags given\n");

  size_t *actual_sizes = calloc (num_svm_pointers, sizeof (size_t));
  if (sizes)
    memcpy (actual_sizes, sizes, num_svm_pointers * sizeof (size_t));
  void **ptrs = malloc (num_svm_pointers * sizeof (void *));
  memcpy (ptrs, svm_pointers, num_svm_pointers * sizeof (void *));

  for (i = 0; i < num_svm_pointers; ++i)
    {
      POCL_RETURN_ERROR_COND ((svm_pointers[i] == NULL), CL_INVALID_VALUE);
      if (sizes && sizes[i]) {
        errcode = pocl_svm_check_pointer (context, svm_pointers[i], sizes[i], NULL);
      } else {
        size_t size = 0;
        void* ptr = NULL;
        errcode = pocl_svm_check_get_pointer (context, svm_pointers[i], 1,
                                              &size, &ptr);
        actual_sizes[i] = size;
        ptrs[i] = ptr;
      }
      if (errcode != CL_SUCCESS)
        goto ERROR;
    }

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  _cl_command_node *cmd = NULL;
  errcode = pocl_create_command (&cmd, command_queue, command_type, event,
                                 num_events_in_wait_list, event_wait_list, 0,
                                 NULL, NULL);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  cmd->command.svm_migrate.svm_pointers = ptrs;
  cmd->command.svm_migrate.num_svm_pointers = num_svm_pointers;
  cmd->command.svm_migrate.sizes = actual_sizes;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;

ERROR:
  POCL_MEM_FREE (cmd);
  POCL_MEM_FREE (ptrs);
  POCL_MEM_FREE (actual_sizes);
  return errcode;
}

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
  return pocl_svm_migrate_mem_common (
      CL_COMMAND_SVM_MIGRATE_MEM, command_queue, num_svm_pointers,
      svm_pointers, sizes, flags, num_events_in_wait_list, event_wait_list,
      event);
}
POsym (clEnqueueSVMMigrateMem)
