/* OpenCL runtime library: clEnqueueNDRangeKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2020 Pekka Jääskeläinen

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
#include "pocl_binary.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_context.h"
#include "pocl_cq_profiling.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "utlist.h"

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif
#include <assert.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

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
  offset_x = offset_y = offset_z = 0;
  global_x = global_y = global_z = 0;
  local_x = local_y = local_z = 0;
  /* cached values for max_work_item_sizes,
   * since we are going to access them repeatedly */
  size_t max_local_x, max_local_y, max_local_z;
  /* cached values for max_work_group_size,
   * since we are going to access them repeatedly */
  size_t max_group_size;

  unsigned i, j;
  int errcode = 0;
  cl_device_id realdev = NULL;
  _cl_command_node *command_node;

  /* no need for malloc, pocl_create_event will memcpy anyway.
   * num_args is the absolute max needed */
  cl_mem *memobj_list
      = (cl_mem *)alloca (kernel->meta->num_args * sizeof (cl_mem));
  size_t memobj_count = 0;
  char *readonly_flag_list
      = (char *)alloca (kernel->meta->num_args * sizeof (char));

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  /* alloc from stack to avoid malloc. num_args is the absolute max needed */
  cl_mem mem_list[kernel->meta->num_args + 1];
  /* reserve space for potential buffer migrate events */
  cl_event new_event_wait_list[num_events_in_wait_list + kernel->meta->num_args
                               + 1];

  POCL_RETURN_ERROR_ON((command_queue->context != kernel->context),
    CL_INVALID_CONTEXT,
    "kernel and command_queue are not from the same context\n");

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_COND((work_dim < 1), CL_INVALID_WORK_DIMENSION);
  POCL_RETURN_ERROR_ON(
    (work_dim > command_queue->device->max_work_item_dimensions),
    CL_INVALID_WORK_DIMENSION,
    "work_dim exceeds devices' max workitem dimensions\n");

  assert (command_queue->device->max_work_item_dimensions <= 3);

  realdev = pocl_real_dev (command_queue->device);

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

  max_local_x = command_queue->device->max_work_item_sizes[0];
  max_local_y
      = work_dim > 1 ? command_queue->device->max_work_item_sizes[1] : 1;
  max_local_z
      = work_dim > 2 ? command_queue->device->max_work_item_sizes[2] : 1;
  max_group_size = command_queue->device->max_work_group_size;

  if (local_work_size != NULL)
    {
      local_x = local_work_size[0];
      local_y = work_dim > 1 ? local_work_size[1] : 1;
      local_z = work_dim > 2 ? local_work_size[2] : 1;

      POCL_RETURN_ERROR_ON((local_x * local_y * local_z > max_group_size),
        CL_INVALID_WORK_GROUP_SIZE,
        "Local worksize dimensions exceed device's max workgroup size\n");

      POCL_RETURN_ERROR_ON((local_x > max_local_x),
        CL_INVALID_WORK_ITEM_SIZE,
        "local_work_size.x > device's max_workitem_sizes[0]\n");

      if (work_dim > 1)
        POCL_RETURN_ERROR_ON((local_y > max_local_y),
          CL_INVALID_WORK_ITEM_SIZE,
          "local_work_size.y > device's max_workitem_sizes[1]\n");

      if (work_dim > 2)
        POCL_RETURN_ERROR_ON((local_z > max_local_z),
          CL_INVALID_WORK_ITEM_SIZE,
          "local_work_size.z > device's max_workitem_sizes[2]\n");

      /* TODO For full 2.x conformance the 'local must divide global'
       * requirement will have to be limited to the cases of kernels compiled
       * with the -cl-uniform-work-group-size option
       */
      POCL_RETURN_ERROR_COND((global_x % local_x != 0),
        CL_INVALID_WORK_GROUP_SIZE);
      POCL_RETURN_ERROR_COND((global_y % local_y != 0),
        CL_INVALID_WORK_GROUP_SIZE);
      POCL_RETURN_ERROR_COND((global_z % local_z != 0),
        CL_INVALID_WORK_GROUP_SIZE);

    }

  /* If the kernel has the reqd_work_group_size attribute, then the local
   * work size _must_ be specified, and it _must_ match the attribute
   * specification
   */
  if (kernel->meta->reqd_wg_size[0] > 0 &&
      kernel->meta->reqd_wg_size[1] > 0 &&
      kernel->meta->reqd_wg_size[2] > 0)
    {
      POCL_RETURN_ERROR_COND ((local_work_size == NULL
                               || local_x != kernel->meta->reqd_wg_size[0]
                               || local_y != kernel->meta->reqd_wg_size[1]
                               || local_z != kernel->meta->reqd_wg_size[2]),
                              CL_INVALID_WORK_GROUP_SIZE);
    }
  /* Otherwise, if the local work size was not specified find the optimal one.
   * Note that at some point we also checked for local > global. This doesn't
   * make sense while we only have 1.2 support for kernel enqueue (and
   * when only uniform group sizes are allowed), but it might turn useful
   * when picking the hardware sub-group size in more sophisticated
   * 2.0 support scenarios.
   */
  else if (local_work_size == NULL)
    {
      if (realdev->ops->compute_local_size)
        realdev->ops->compute_local_size (realdev, global_x, global_y,
                                          global_z, &local_x, &local_y,
                                          &local_z);
      else
        pocl_default_local_size_optimizer (realdev, global_x, global_y,
                                           global_z, &local_x, &local_y,
                                           &local_z);
    }

  POCL_MSG_PRINT_INFO("Queueing kernel %s with local size %u x %u x %u group "
                      "sizes %u x %u x %u...\n",
                      kernel->name,
                      (unsigned)local_x, (unsigned)local_y, (unsigned)local_z,
                      (unsigned)(global_x / local_x),
                      (unsigned)(global_y / local_y),
                      (unsigned)(global_z / local_z));

  assert (local_x * local_y * local_z <= max_group_size);
  assert (local_x <= max_local_x);
  assert (local_y <= max_local_y);
  assert (local_z <= max_local_z);

  /* See TODO above for 'local must divide global' */
  assert (global_x % local_x == 0);
  assert (global_y % local_y == 0);
  assert (global_z % local_z == 0);

  for (i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument_info *a = &kernel->meta->arg_info[i];
      struct pocl_argument *al = &(kernel->dyn_arguments[i]);

      POCL_RETURN_ERROR_ON ((!al->is_set), CL_INVALID_KERNEL_ARGS,
                            "The %i-th kernel argument is not set!\n", i);

      if (a->type == POCL_ARG_TYPE_IMAGE
          || (!ARGP_IS_LOCAL (a) && a->type == POCL_ARG_TYPE_POINTER
              && al->value != NULL && al->is_svm == 0))
        {
          cl_mem buf = *(cl_mem *) (al->value);

          if (a->type == POCL_ARG_TYPE_IMAGE)
            {
              POCL_RETURN_ON_UNSUPPORTED_IMAGE (buf, realdev);
            }
          else
            {
              /* subbuffers are handled in clSetKernelArg */
              assert (buf->parent == NULL);

              if (al->offset > 0)
                POCL_RETURN_ERROR_ON (
                    (al->offset % realdev->mem_base_addr_align != 0),
                    CL_MISALIGNED_SUB_BUFFER_OFFSET,
                    "SubBuffer is not properly aligned for this device");

              POCL_RETURN_ERROR_ON (
                  (buf->size > realdev->max_mem_alloc_size),
                  CL_OUT_OF_RESOURCES,
                  "ARG %u: buffer is larger than "
                  "device's MAX_MEM_ALLOC_SIZE\n",
                  i);
            }

          if (al->is_readonly || (buf->flags & CL_MEM_READ_ONLY))
            {
              if (al->is_readonly == 0)
                POCL_MSG_WARN ("readonly buffer used as kernel arg, but arg "
                               "type is not const\n");
              readonly_flag_list[memobj_count] = 1;
            }
          else
            readonly_flag_list[memobj_count] = 0;

          memobj_list[memobj_count++] = buf;
        }
    }

  errcode = pocl_create_command (
      &command_node, command_queue, CL_COMMAND_NDRANGE_KERNEL, event,
      num_events_in_wait_list, event_wait_list, memobj_count, memobj_list,
      readonly_flag_list);

  if (errcode != CL_SUCCESS)
    {
      POCL_MSG_ERR ("Failed to create command: %i\n", errcode);
      return errcode;
    }

  command_node->type = CL_COMMAND_NDRANGE_KERNEL;
  command_node->command.run.kernel = kernel;
  command_node->command.run.pc.local_size[0] = local_x;
  command_node->command.run.pc.local_size[1] = local_y;
  command_node->command.run.pc.local_size[2] = local_z;
  command_node->command.run.pc.work_dim = work_dim;
  command_node->command.run.pc.num_groups[0] = global_x / local_x;
  command_node->command.run.pc.num_groups[1] = global_y / local_y;
  command_node->command.run.pc.num_groups[2] = global_z / local_z;
  command_node->command.run.pc.global_offset[0] = offset_x;
  command_node->command.run.pc.global_offset[1] = offset_y;
  command_node->command.run.pc.global_offset[2] = offset_z;

  cl_uint program_dev_i = CL_UINT_MAX;
  for (i = 0; i < kernel->program->num_devices; ++i)
    {
      if (kernel->program->devices[i] == realdev)
        program_dev_i = i;
    }
  assert (program_dev_i < CL_UINT_MAX);
  command_node->device_i = program_dev_i;
  command_node->command.run.hash = kernel->meta->build_hash[program_dev_i];

  /* Copy the currently set kernel arguments because the same kernel
     object can be reused for new launches with different arguments. */
  command_node->command.run.arguments =
    (struct pocl_argument *) malloc ((kernel->meta->num_args) *
                                     sizeof (struct pocl_argument));

  for (i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument *arg = &command_node->command.run.arguments[i];
      memcpy (arg, &kernel->dyn_arguments[i], sizeof (pocl_argument));

      if (arg->value != NULL)
        {
          size_t arg_alloc_size = arg->size;
          assert (arg_alloc_size > 0);
          /* FIXME: this is a cludge to determine an acceptable alignment,
           * we should probably extract the argument alignment from the
           * LLVM bytecode during kernel header generation. */
          size_t arg_alignment = pocl_size_ceil2 (arg_alloc_size);
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

  if (pocl_cq_profiling_enabled)
    {
      pocl_cq_profiling_register_event (command_node->event);
      POname(clRetainKernel) (kernel);
      command_node->event->meta_data->kernel = kernel;
    }

  pocl_command_enqueue (command_queue, command_node);
  return CL_SUCCESS;
}
POsym(clEnqueueNDRangeKernel)
