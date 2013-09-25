#include "config.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "string.h"
#include "pocl_cl.h"
#include "utlist.h"

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
  _cl_command_node *command_node;
  cl_mem *mem_list_copy;
  void *args_copy;
  int error;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (user_func == NULL)
    return CL_INVALID_VALUE;

  if (args == NULL && (cb_args > 0 || num_mem_objects > 0))
    return CL_INVALID_VALUE;

  if (args != NULL && cb_args == 0)
    return CL_INVALID_VALUE;

  if (num_mem_objects > 0 && (mem_list == NULL || args_mem_loc == NULL))
    return CL_INVALID_VALUE;

  if (num_mem_objects == 0 && (mem_list != NULL || args_mem_loc != NULL))
    return CL_INVALID_VALUE;

  if (!command_queue->device->execution_capabilities & CL_EXEC_NATIVE_KERNEL)
    return CL_INVALID_OPERATION;

  error = pocl_create_command (&command_node, command_queue,
                               CL_COMMAND_NATIVE_KERNEL,
                               event, num_events_in_wait_list,
                               event_wait_list);
  if (error != CL_SUCCESS)
    return error;

  command_node->command.native.data = command_queue->device->data;
  command_node->command.native.num_mem_objects = num_mem_objects;
  command_node->command.native.user_func = user_func;

  if (event != NULL)
    {
      error = pocl_create_event (event, command_queue,
                                 CL_COMMAND_NATIVE_KERNEL);
      if (error != CL_SUCCESS)
        return error;
      POCL_UPDATE_EVENT_QUEUED;
    }

  /* Specification specifies that args passed to user_func is a copy of the
   * one passed to this function */
  args_copy = malloc(cb_args);
  if (args_copy == NULL)
    {
      free(command_node);
      return CL_OUT_OF_HOST_MEMORY;
    }
  memcpy(args_copy, args, cb_args);

  /* recopy the cl_mem object list to free them easily after run */
  mem_list_copy = malloc(num_mem_objects * sizeof(cl_mem));
  if (mem_list_copy == NULL)
    {
      free(args_copy);
      free(command_node);
      return CL_OUT_OF_HOST_MEMORY;
    }
  memcpy(mem_list_copy, mem_list, num_mem_objects * sizeof(cl_mem));
  command_node->command.native.mem_list = mem_list_copy;

  for (i = 0; i < num_mem_objects; i++)
    {
      void *buf;
      const char *loc = (const char *) args_mem_loc[i];
      void *arg_loc;

      if (mem_list[i] == NULL)
        {
          free(args_copy);
          free(mem_list_copy);
          free(command_node);
          return CL_INVALID_MEM_OBJECT;
        }

      /* put the device ptr of the clmem in the argument */
      buf = mem_list[i]->device_ptrs[command_queue->device->dev_id];

      POname(clRetainMemObject) (mem_list[i]);
      /* args_mem_loc is a pointer relative to the original args, since we
       * recopy them, we must do some relocation */
      off_t offset = (uintptr_t) loc - (uintptr_t) args;

      arg_loc = (void *) ((uintptr_t) args_copy + (uintptr_t)offset);

      if (command_queue->device->address_bits == 32)
          *((uint32_t *) arg_loc) = (uint32_t) (((uintptr_t) buf) & 0xFFFFFFFF);
      else
          *((uint64_t *) arg_loc) = (uint64_t) (uintptr_t) buf;
    }
  command_node->command.native.args = args_copy;
  command_node->command.native.cb_args = cb_args;

  POname(clRetainCommandQueue) (command_queue);

  LL_APPEND(command_queue->root, command_node);

  return CL_SUCCESS;
}
POsym(clEnqueueNativeKernel)
