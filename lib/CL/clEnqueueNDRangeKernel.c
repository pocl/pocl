/* OpenCL runtime library: clEnqueueNDRangeKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2013 Pekka Jääskeläinen / Tampere University of Technology

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

#include "config.h"
#include "pocl_cl.h"
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "pocl_cache.h"
#include "utlist.h"
#include "pocl_binary.h"
#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif
#include <assert.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

#define COMMAND_LENGTH 1024
#define ARGUMENT_STRING_LENGTH 32

//#define DEBUG_NDRANGE

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueNDRangeKernel)(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t *global_work_offset,
                       const size_t *global_work_size,
                       const size_t *local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  size_t offset_x, offset_y, offset_z;
  size_t global_x, global_y, global_z;
  size_t local_x, local_y, local_z;
  int b_migrate_count, buffer_count;
  unsigned i;
  int error = 0;
  cl_device_id realdev = NULL;
  struct pocl_context pc;
  _cl_command_node *command_node;
  /* alloc from stack to avoid malloc. num_args is the absolute max needed */
  cl_mem mem_list[kernel->num_args];
  /* reserve space for potential buffer migrate events */
  cl_event new_event_wait_list[num_events_in_wait_list + kernel->num_args];

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON((command_queue->context != kernel->context),
    CL_INVALID_CONTEXT, "kernel and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_COND((work_dim < 1), CL_INVALID_WORK_DIMENSION);
  POCL_RETURN_ERROR_ON((work_dim > command_queue->device->max_work_item_dimensions),
    CL_INVALID_WORK_DIMENSION, "work_dim exceeds devices' max workitem dimensions\n");

  assert(command_queue->device->max_work_item_dimensions <= 3);

  realdev = POCL_REAL_DEV(command_queue->device);

  if (global_work_offset != NULL)
    {
      offset_x = global_work_offset[0];
      offset_y = work_dim > 1 ? global_work_offset[1] : 0;
      offset_z = work_dim > 2 ? global_work_offset[2] : 0;
    }
  else
    {
      offset_x = 0;
      offset_y = 0;
      offset_z = 0;
    }

  global_x = global_work_size[0];
  global_y = work_dim > 1 ? global_work_size[1] : 1;
  global_z = work_dim > 2 ? global_work_size[2] : 1;

  POCL_RETURN_ERROR_COND((global_x == 0 || global_y == 0 || global_z == 0),
    CL_INVALID_GLOBAL_WORK_SIZE);

  for (i = 0; i < kernel->num_args; i++)
    {
      POCL_RETURN_ERROR_ON((!kernel->arg_info[i].is_set), CL_INVALID_KERNEL_ARGS,
        "The %i-th kernel argument is not set!\n", i);
    }

  if (local_work_size != NULL)
    {
      local_x = local_work_size[0];
      local_y = work_dim > 1 ? local_work_size[1] : 1;
      local_z = work_dim > 2 ? local_work_size[2] : 1;
    }

  /* If the kernel has the reqd_work_group_size attribute, then the local
   * work size _must_ be specified, and it _must_ match the attribute specification
   */
  if (kernel->reqd_wg_size != NULL &&
      kernel->reqd_wg_size[0] > 0 &&
      kernel->reqd_wg_size[1] > 0 &&
      kernel->reqd_wg_size[2] > 0)
    {
      POCL_RETURN_ERROR_COND((local_work_size == NULL ||
          local_x != kernel->reqd_wg_size[0] ||
          local_y != kernel->reqd_wg_size[1] ||
          local_z != kernel->reqd_wg_size[2]), CL_INVALID_WORK_GROUP_SIZE);
    }
  /* otherwise, if the local work size was not specified or it's bigger
   * than the global work size, find the optimal one
   */
  else if (local_work_size == NULL ||
      (local_x > global_x || local_y > global_y || local_z > global_z))
    {
      /* Embarrassingly parallel kernel with a free work-group
         size. Try to figure out one which utilizes all the
         resources efficiently. Assume work-groups are scheduled
         to compute units, so try to split it to a number of
         work groups at the equal to the number of CUs, while still
         trying to respect the preferred WG size multiple (for better
         SIMD instruction utilization).
      */
      size_t preferred_wg_multiple;
      POname(clGetKernelWorkGroupInfo)
        (kernel, command_queue->device,
         CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
         sizeof (size_t), &preferred_wg_multiple, NULL);

      POCL_MSG_PRINT_INFO("Preferred WG size multiple %zu\n",
                          preferred_wg_multiple);

      local_x = global_x;
      local_y = global_y;
      local_z = global_z;

      /* First try to split a dimension with the WG multiple
         to make it still be divisible with the WG multiple. */
      do {
        /* Split the dimension, but avoid ending up with a dimension that
           is not multiple of the wanted size. */
        if (local_x > 1 && local_x % 2 == 0 &&
            (local_x / 2) % preferred_wg_multiple == 0)
          {
            local_x /= 2;
            continue;
          }
        else if (local_y > 1 && local_y % 2 == 0 &&
                 (local_y / 2) % preferred_wg_multiple == 0)
          {
            local_y /= 2;
            continue;
          }
        else if (local_z > 1 && local_z % 2 == 0 &&
                 (local_z / 2) % preferred_wg_multiple == 0)
          {
            local_z /= 2;
            continue;
          }

        /* Next find out a dimension that is not a multiple anyways,
           so one cannot nicely vectorize over it, and set it to one. */
        if (local_z > 1 && local_z % preferred_wg_multiple != 0)
          {
            local_z = 1;
            continue;
          }
        else if (local_y > 1 && local_y % preferred_wg_multiple != 0)
          {
            local_y = 1;
            continue;
          }
        else if (local_z > 1 && local_z % preferred_wg_multiple != 0)
          {
            local_z = 1;
            continue;
          }

        /* Finally, start setting them to zero starting from the Z
           dimension. */
        if (local_z > 1)
          {
            local_z = 1;
            continue;
          }
        else if (local_y > 1)
          {
            local_y = 1;
            continue;
          }
        else if (local_x > 1)
          {
            local_x = 1;
            continue;
          }
      }
      while (local_x * local_y * local_z >
             command_queue->device->max_work_group_size);
    }

  POCL_MSG_PRINT_INFO("Queueing kernel %s with local size %u x %u x %u group "
                      "sizes %u x %u x %u...\n",
                      kernel->name,
                      (unsigned)local_x, (unsigned)local_y, (unsigned)local_z,
                      (unsigned)(global_x / local_x),
                      (unsigned)(global_y / local_y),
                      (unsigned)(global_z / local_z));

  POCL_RETURN_ERROR_ON((local_x * local_y * local_z > command_queue->device->max_work_group_size),
    CL_INVALID_WORK_GROUP_SIZE, "Local worksize dimensions exceed device's max workgroup size\n");

  POCL_RETURN_ERROR_ON((local_x > command_queue->device->max_work_item_sizes[0]),
    CL_INVALID_WORK_ITEM_SIZE, "local_work_size.x > device's max_workitem_sizes[0]\n");

  if (work_dim > 1)
    POCL_RETURN_ERROR_ON((local_y > command_queue->device->max_work_item_sizes[1]),
    CL_INVALID_WORK_ITEM_SIZE, "local_work_size.y > device's max_workitem_sizes[1]\n");

  if (work_dim > 2)
    POCL_RETURN_ERROR_ON((local_z > command_queue->device->max_work_item_sizes[2]),
    CL_INVALID_WORK_ITEM_SIZE, "local_work_size.z > device's max_workitem_sizes[2]\n");

  POCL_RETURN_ERROR_COND((global_x % local_x != 0), CL_INVALID_WORK_GROUP_SIZE);
  POCL_RETURN_ERROR_COND((global_y % local_y != 0), CL_INVALID_WORK_GROUP_SIZE);
  POCL_RETURN_ERROR_COND((global_z % local_z != 0), CL_INVALID_WORK_GROUP_SIZE);

  POCL_RETURN_ERROR_COND((event_wait_list == NULL && num_events_in_wait_list > 0),
    CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND((event_wait_list != NULL && num_events_in_wait_list == 0),
    CL_INVALID_EVENT_WAIT_LIST);

  char cachedir[POCL_FILENAME_LENGTH];
  int realdev_i = pocl_cl_device_to_index (kernel->program, realdev);
  assert(realdev_i >= 0);
  pocl_cache_kernel_cachedir_path (cachedir, kernel->program,
                                   realdev_i, kernel, "",
                                   local_x, local_y, local_z);

  if (kernel->program->source || kernel->program->binaries[realdev_i])
    {
#ifdef OCS_AVAILABLE
      // SPMD devices already have compiled at this point
      if (realdev->spmd)
        error = CL_SUCCESS;
      else
        error = pocl_llvm_generate_workgroup_function (cachedir, realdev, kernel,
                                                       local_x, local_y, local_z);
#else
      error = 1;
#endif
      if (error) goto ERROR;
    }

  b_migrate_count = 0;
  buffer_count = 0;

  /* count mem objects and enqueue needed mem migrations */
  for (i = 0; i < kernel->num_args; ++i)
    {
      struct pocl_argument *al = &(kernel->dyn_arguments[i]);
      if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
          (!kernel->arg_info[i].is_local
           && kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER
           && al->value != NULL))
        {
          cl_mem buf = *(cl_mem *) (al->value);
          mem_list[buffer_count++] = buf;
          POname(clRetainMemObject) (buf);
          /* if buffer has no owner,
             it has not been used yet -> just claim it */
          if (buf->owning_device == NULL)
            buf->owning_device = realdev;
          /* If buffer is located located in another global memory
             (other device), it needs to be migrated before this kernel
             may be executed */
          if (buf->owning_device != NULL &&
              buf->owning_device->global_mem_id !=
              command_queue->device->global_mem_id)
            {
#if DEBUG_NDRANGE
              printf("mem migrate needed: owning dev = %d, target dev = %d\n",
                     buf->owning_device->global_mem_id,
                     command_queue->device->global_mem_id);
#endif
              cl_event mem_event = buf->latest_event;
              POname(clEnqueueMigrateMemObjects)
                (command_queue, 1, &buf, 0, (mem_event ? 1 : 0),
                 (mem_event ? &mem_event : NULL),
                 &new_event_wait_list[b_migrate_count++]);
            }
          buf->owning_device = realdev;
        }
    }

  if (num_events_in_wait_list)
    {
      memcpy (&new_event_wait_list[b_migrate_count], event_wait_list,
              sizeof(cl_event) * num_events_in_wait_list);
    }

  error = pocl_create_command (&command_node, command_queue,
                               CL_COMMAND_NDRANGE_KERNEL, event,
                               num_events_in_wait_list + b_migrate_count,
                               (num_events_in_wait_list + b_migrate_count)?
                               new_event_wait_list : NULL,
                               buffer_count, mem_list);
  if (error != CL_SUCCESS)
    goto ERROR;

  pc.work_dim = work_dim;
  pc.num_groups[0] = global_x / local_x;
  pc.num_groups[1] = global_y / local_y;
  pc.num_groups[2] = global_z / local_z;
  pc.global_offset[0] = offset_x;
  pc.global_offset[1] = offset_y;
  pc.global_offset[2] = offset_z;

  command_node->type = CL_COMMAND_NDRANGE_KERNEL;
  command_node->command.run.data = command_queue->device->data;
  command_node->command.run.tmp_dir = strdup (cachedir);
  command_node->command.run.kernel = kernel;
  command_node->command.run.pc = pc;
  command_node->command.run.local_x = local_x;
  command_node->command.run.local_y = local_y;
  command_node->command.run.local_z = local_z;

  /* Copy the currently set kernel arguments because the same kernel
     object can be reused for new launches with different arguments. */
  command_node->command.run.arguments =
    (struct pocl_argument *) malloc ((kernel->num_args + kernel->num_locals) *
                                     sizeof (struct pocl_argument));

  for (i = 0; i < kernel->num_args + kernel->num_locals; ++i)
    {
      struct pocl_argument *arg = &command_node->command.run.arguments[i];
      size_t arg_alloc_size = kernel->dyn_arguments[i].size;
      arg->size = arg_alloc_size;

      if (kernel->dyn_arguments[i].value == NULL)
        {
          arg->value = NULL;
        }
      else
        {
          /* FIXME: this is a cludge to determine an acceptable alignment,
           * we should probably extract the argument alignment from the
           * LLVM bytecode during kernel header generation. */
          size_t arg_alignment = pocl_size_ceil2(arg_alloc_size);
          if (arg_alignment >= MAX_EXTENDED_ALIGNMENT)
            arg_alignment = MAX_EXTENDED_ALIGNMENT;
          if (arg_alloc_size < arg_alignment)
            arg_alloc_size = arg_alignment;

          arg->value = pocl_aligned_malloc (arg_alignment, arg_alloc_size);
          memcpy (arg->value, kernel->dyn_arguments[i].value, arg->size);
        }
    }

  command_node->next = NULL;

  POname(clRetainKernel) (kernel);

  pocl_command_enqueue (command_queue, command_node);
  error = CL_SUCCESS;

ERROR:
  return error;

}
POsym(clEnqueueNDRangeKernel)
