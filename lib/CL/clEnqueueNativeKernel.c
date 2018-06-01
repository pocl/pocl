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

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

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


  errcode = pocl_create_command (&command_node, command_queue,
                               CL_COMMAND_NATIVE_KERNEL,
                               event, num_events_in_wait_list,
                               event_wait_list, num_mem_objects, mem_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  command_node->command.native.user_func = user_func;

  /* Specification specifies that args passed to user_func is a copy of the
   * one passed to this function */
  if (cb_args)
    {
      args_copy = malloc (cb_args);
      if (args_copy == NULL)
      {
        POCL_MEM_FREE(command_node);
        return CL_OUT_OF_HOST_MEMORY;
      }
      memcpy (args_copy, args, cb_args);
    }

  for (i = 0; i < num_mem_objects; i++)
    {
      void *buf;
      const char *loc = (const char *) args_mem_loc[i];
      void *arg_loc;

      if (mem_list[i] == NULL)
        {
          POCL_MEM_FREE(args_copy);
          POCL_MEM_FREE(command_node);
          return CL_INVALID_MEM_OBJECT;
        }

      /* put the device ptr of the clmem in the argument */
      buf = mem_list[i]->device_ptrs[command_queue->device->dev_id].mem_ptr;
      mem_list[i]->owning_device = command_queue->device;
      POname(clRetainMemObject) (mem_list[i]);
      /* args_mem_loc is a pointer relative to the original args, since we
       * recopy them, we must do some relocation */
      ptrdiff_t offset = (uintptr_t) loc - (uintptr_t) args;

      arg_loc = (void *) ((uintptr_t) args_copy + (uintptr_t)offset);

      if (command_queue->device->address_bits == 32)
          *((uint32_t *) arg_loc) = (uint32_t) (((uintptr_t) buf) & 0xFFFFFFFF);
      else
          *((uint64_t *) arg_loc) = (uint64_t) (uintptr_t) buf;
    }
  command_node->command.native.args = args_copy;
  command_node->command.native.cb_args = cb_args;

  pocl_command_enqueue (command_queue, command_node);

  return CL_SUCCESS;
}
POsym(clEnqueueNativeKernel)
