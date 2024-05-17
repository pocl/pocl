/* OpenCL runtime library: clEnqueueNativeKernel()

   Copyright (c) 2010-2023 PoCL developers

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

#include "config.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "string.h"
#include "pocl_cl.h"
#include "utlist.h"

#include <limits.h>

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueNativeKernel)(cl_command_queue   command_queue ,
					  void (CL_CALLBACK * user_func)(void *), 
                      void *             args ,
                      size_t             cb_args , 
                      cl_uint            num_mem_objects ,
                      const cl_mem *     mem_list ,
                      const void **      args_mem_loc ,
                      cl_uint            num_events_in_wait_list ,
                      const cl_event *   event_wait_list ,
                      cl_event *         event ) CL_API_SUFFIX__VERSION_1_0
{
  cl_uint i = 0;
  _cl_command_node *command_node = NULL;
  void *args_copy = NULL;
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  POCL_RETURN_ERROR_COND((user_func == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND(((args == NULL) && (cb_args > 0 )), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND(((args == NULL) && (num_mem_objects > 0)), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND(((args != NULL) && (cb_args == 0)), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND(((num_mem_objects > 0) && (mem_list == NULL)), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND(((num_mem_objects > 0) && (args_mem_loc == NULL)), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND(((num_mem_objects == 0) && (mem_list != NULL)), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND(((num_mem_objects == 0) && (args_mem_loc != NULL)), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON(!(command_queue->device->execution_capabilities &
    CL_EXEC_NATIVE_KERNEL), CL_INVALID_OPERATION, "device associated with "
    "command_queue cannot execute the native kernel\n");

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_buffer_migration_info *migr_infos = NULL;
  char *rdonly = (char *)alloca (num_mem_objects);
  cl_mem *ml = (cl_mem *)alloca (num_mem_objects * sizeof (cl_mem));
  memcpy (ml, mem_list, num_mem_objects * sizeof (cl_mem));

  for (i = 0; i < num_mem_objects; i++)
    {
      POCL_RETURN_ERROR_ON ((!IS_CL_OBJECT_VALID (ml[i])),
                            CL_INVALID_MEM_OBJECT,
                            "The %i-th mem object is invalid\n", i);

      migr_infos = pocl_append_unique_migration_info (
        migr_infos, ml[i], !!(ml[i]->flags & CL_MEM_READ_ONLY));
    }

  /* Specification specifies that args passed to user_func is a copy of the
   * one passed to this function */
  if (cb_args)
    {
      args_copy = malloc (cb_args);
      POCL_RETURN_ERROR_COND ((args_copy == NULL), CL_OUT_OF_HOST_MEMORY);
      memcpy (args_copy, args, cb_args);
    }

  void **arg_locs = (void **)calloc (num_mem_objects, sizeof (void *));
  POCL_RETURN_ERROR_COND ((arg_locs == NULL), CL_OUT_OF_HOST_MEMORY);

  for (i = 0; i < num_mem_objects; i++)
    {
      void *arg_loc;

      /* args_mem_loc is a pointer relative to the original args;
       * since we recopy them, must also relocate them against copy.
       * Note that we cannot use device_ptr pointers here directly,
       * because of lazy memory allocation/migration. */
      assert ((uintptr_t)args_mem_loc[i] >= (uintptr_t)args);
      uintptr_t offset = (uintptr_t)args_mem_loc[i] - (uintptr_t)args;
      arg_loc = (void *)((uintptr_t)args_copy + offset);
      arg_locs[i] = arg_loc;
    }

  errcode = pocl_create_command (
    &command_node, command_queue, CL_COMMAND_NATIVE_KERNEL, event,
    num_events_in_wait_list, event_wait_list, migr_infos);

  if (errcode != CL_SUCCESS)
    return errcode;

  command_node->command.native.user_func = user_func;
  command_node->command.native.arg_locs = arg_locs;
  command_node->command.native.args = args_copy;
  command_node->command.native.cb_args = cb_args;

  pocl_command_enqueue (command_queue, command_node);

  return CL_SUCCESS;
}
POsym(clEnqueueNativeKernel)
