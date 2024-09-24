/* pocl_ndrange_kernel.c: helpers for NDRange Kernel commands

   Copyright (c) 2022-2024 Jan Solanti / Tampere University
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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
#include "pocl_local_size.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

#include <assert.h>

static cl_int
pocl_kernel_calc_wg_size (cl_command_queue command_queue, cl_kernel kernel,
                          unsigned device_i,
                          cl_uint work_dim, const size_t *global_work_offset,
                          const size_t *global_work_size,
                          const size_t *local_work_size, size_t *global_offset,
                          size_t *local_size, size_t *num_groups)
{
  size_t offset_x, offset_y, offset_z;
  size_t global_x, global_y, global_z;
  size_t local_x, local_y, local_z;
  offset_x = offset_y = offset_z = 0;
  global_x = global_y = global_z = 0;
  local_x = local_y = local_z = 1;
  /* cached values for max_work_item_sizes,
   * since we are going to access them repeatedly */
  size_t max_local_x, max_local_y, max_local_z;
  /* cached values for max_work_group_size,
   * since we are going to access them repeatedly */
  size_t max_group_size;

  POCL_RETURN_ERROR_COND ((work_dim < 1), CL_INVALID_WORK_DIMENSION);
  POCL_RETURN_ERROR_ON (
      (work_dim > command_queue->device->max_work_item_dimensions),
      CL_INVALID_WORK_DIMENSION,
      "work_dim exceeds devices' max workitem dimensions\n");

  assert (command_queue->device->max_work_item_dimensions <= 3);

  cl_device_id realdev = pocl_real_dev (command_queue->device);

  if (global_work_size == NULL)
    {
      global_x = global_y = global_z = 0;
      goto SKIP_WG_SIZE_CALCULATION;
    }

  cl_uint zero_gws = 0;
  for (cl_uint i = 0; i < work_dim; ++i)
    {
      zero_gws += (global_work_size[i] == 0);
    }
  if (zero_gws)
    {
      global_x = global_y = global_z = 0;
      goto SKIP_WG_SIZE_CALCULATION;
    }

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

  if (global_x == 0 || global_y == 0 || global_z == 0)
    {
      global_x = global_y = global_z = 0;
      goto SKIP_WG_SIZE_CALCULATION;
    }

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

      POCL_RETURN_ERROR_ON (
          (local_x * local_y * local_z > max_group_size),
          CL_INVALID_WORK_GROUP_SIZE,
          "Local worksize dimensions exceed device's max workgroup size\n");

      POCL_RETURN_ERROR_ON (
          (local_x > max_local_x), CL_INVALID_WORK_ITEM_SIZE,
          "local_work_size.x > device's max_workitem_sizes[0]\n");

      if (work_dim > 1)
        POCL_RETURN_ERROR_ON (
            (local_y > max_local_y), CL_INVALID_WORK_ITEM_SIZE,
            "local_work_size.y > device's max_workitem_sizes[1]\n");

      if (work_dim > 2)
        POCL_RETURN_ERROR_ON (
            (local_z > max_local_z), CL_INVALID_WORK_ITEM_SIZE,
            "local_work_size.z > device's max_workitem_sizes[2]\n");

      /* TODO For full 2.x conformance the 'local must divide global'
       * requirement will have to be limited to the cases of kernels compiled
       * with the -cl-uniform-work-group-size option
       */
      POCL_RETURN_ERROR_COND ((global_x % local_x != 0),
                              CL_INVALID_WORK_GROUP_SIZE);
      POCL_RETURN_ERROR_COND ((global_y % local_y != 0),
                              CL_INVALID_WORK_GROUP_SIZE);
      POCL_RETURN_ERROR_COND ((global_z % local_z != 0),
                              CL_INVALID_WORK_GROUP_SIZE);
    }

  if (realdev->ops->verify_ndrange_sizes)
    {
      // verify the sanitized NDRange values
      size_t OFS[3] = { offset_x, offset_y, offset_z };
      size_t GWS[3] = { global_x, global_y, global_z };
      size_t LWS[3] = { local_x, local_y, local_z };

      int errcode = realdev->ops->verify_ndrange_sizes (OFS, GWS, LWS);
      if (errcode != CL_SUCCESS)
        return errcode;
    }

  /* If the kernel has the reqd_work_group_size attribute, then the local
   * work size _must_ be specified, and it _must_ match the attribute
   * specification
   */
  if (kernel->meta->reqd_wg_size[0] > 0 && kernel->meta->reqd_wg_size[1] > 0
      && kernel->meta->reqd_wg_size[2] > 0)
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
        realdev->ops->compute_local_size (realdev, kernel, device_i,
                                          global_x, global_y,
                                          global_z, &local_x, &local_y,
                                          &local_z);
      else
        pocl_default_local_size_optimizer (realdev, kernel, device_i,
                                           global_x, global_y,
                                           global_z, &local_x, &local_y,
                                           &local_z);
    }

  POCL_MSG_PRINT_INFO (
      "Preparing kernel %s with local size %u x %u x %u group "
      "sizes %u x %u x %u...\n",
      kernel->name, (unsigned)local_x, (unsigned)local_y, (unsigned)local_z,
      (unsigned)(global_x / local_x), (unsigned)(global_y / local_y),
      (unsigned)(global_z / local_z));

  assert (local_x * local_y * local_z <= max_group_size);
  assert (local_x <= max_local_x);
  assert (local_y <= max_local_y);
  assert (local_z <= max_local_z);

  /* See TODO above for 'local must divide global' */
  assert (global_x % local_x == 0);
  assert (global_y % local_y == 0);
  assert (global_z % local_z == 0);

SKIP_WG_SIZE_CALCULATION:
  local_size[0] = local_x;
  local_size[1] = local_y;
  local_size[2] = local_z;
  global_offset[0] = offset_x;
  global_offset[1] = offset_y;
  global_offset[2] = offset_z;
  num_groups[0] = global_x / local_x;
  num_groups[1] = global_y / local_y;
  num_groups[2] = global_z / local_z;

  return CL_SUCCESS;
}

/**
 * Collect the kernel's buffer usage for implicit migration.
 */
static cl_int
pocl_kernel_collect_mem_objs (
  cl_command_queue command_queue,
  cl_kernel kernel,
  struct pocl_argument *src_arguments,
  pocl_buffer_migration_info **dst_migr_infos)
{
  cl_device_id realdev = pocl_real_dev (command_queue->device);
  pocl_buffer_migration_info *migr_infos = NULL;

  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument_info *a = &kernel->meta->arg_info[i];
      struct pocl_argument *al = &(src_arguments[i]);

      POCL_RETURN_ERROR_ON ((!al->is_set), CL_INVALID_KERNEL_ARGS,
                            "The %i-th kernel argument is not set!\n", i);

      if (a->type == POCL_ARG_TYPE_IMAGE || a->type == POCL_ARG_TYPE_PIPE
          || (!ARGP_IS_LOCAL (a) && a->type == POCL_ARG_TYPE_POINTER
              && al->value != NULL))
        {
          cl_mem buf = *(cl_mem *)(al->value);
          if (al->is_raw_ptr)
            {
              /* Find the shadow cl_mem wrapper which is used for tracking
                 migrations. */
              void *ptr = *(void **)(al->value);
              pocl_raw_ptr *raw_ptr = pocl_find_raw_ptr_with_vm_ptr (
                  command_queue->context, ptr);

              /* TODO: The case where some of the args are SVM-allocated VM
                 pointers, some device pointers. These address spaces are
                 allowed to overlap and we could have the SVM and dev buffer
                 having the same addr in theory. */
              if (raw_ptr == NULL)
                raw_ptr = pocl_find_raw_ptr_with_dev_ptr (
                    command_queue->context, ptr);
              if (raw_ptr == NULL)
                {
                  POCL_MSG_PRINT_MEMORY (
                      "Couldn't find the shadow cl_mem for an SVM ptr, "
                      "assuming system SVM.\n");
                  continue;
                }
              else
                buf = raw_ptr->shadow_cl_mem;
            }

          if (buf == NULL) /* Likely USM. */
            continue;
          if (a->type == POCL_ARG_TYPE_IMAGE)
            {
              POCL_RETURN_ON_UNSUPPORTED_IMAGE (buf, realdev);
            }
          else
            {
              if (buf->origin > 0)
                POCL_RETURN_ERROR_ON (
                  (!buf->has_device_address
                   && buf->origin % realdev->mem_base_addr_align != 0),
                  CL_MISALIGNED_SUB_BUFFER_OFFSET,
                  "SubBuffer is not properly aligned for this device");

              POCL_RETURN_ERROR_ON ((buf->size > realdev->max_mem_alloc_size),
                                    CL_OUT_OF_RESOURCES,
                                    "ARG %u: buffer is larger (%lu) than "
                                    "device's MAX_MEM_ALLOC_SIZE (%lu).\n",
                                    i, buf->size, realdev->max_mem_alloc_size);
            }

          char read_only = 0;
          if (al->is_readonly || (buf->flags & CL_MEM_READ_ONLY))
            {
              if (al->is_readonly == 0)
                POCL_MSG_WARN ("readonly buffer used as kernel arg, but arg "
                               "type is not const\n");
              read_only = 1;
            }
          migr_infos
            = pocl_append_unique_migration_info (migr_infos, buf, read_only);
        }
    }

  /* If the kernel has a general indirect access set, we append the
     currently-alive raw buffers to the indirect_raw_buffers set and ensure
     their data gets synchronized to the device. */
  if (kernel->can_access_all_raw_buffers_indirectly)
    {
      struct _pocl_raw_ptr *ptr;
      DL_FOREACH (kernel->context->raw_ptrs, ptr)
      {
        migr_infos = pocl_append_unique_migration_info (migr_infos,
                                                        ptr->shadow_cl_mem, 0);
      }
    }
  else
    {
      /* Otherwise, we only migrate buffers related to USM/SVM pointers
         explicitly set with clSetKernelExecInfo(). */
      struct _pocl_ptr_list_node *n;
      DL_FOREACH (kernel->indirect_raw_ptrs, n)
      {
        pocl_raw_ptr *svm_ptr
          = pocl_find_raw_ptr_with_vm_ptr (command_queue->context, n->ptr);

        if (svm_ptr == NULL)
          {
            POCL_MSG_PRINT_MEMORY ("Couldn't find the shadow cl_mem for an "
                                   "clSetKernelExecInfo-set SVM ptr, "
                                   "assuming system SVM.\n");
            continue;
          }

        migr_infos = pocl_append_unique_migration_info (
          migr_infos, svm_ptr->shadow_cl_mem, 0);
      }
    }
  *dst_migr_infos = migr_infos;
  return CL_SUCCESS;
}

cl_int
pocl_kernel_copy_args (cl_kernel kernel,
                       struct pocl_argument *src_arguments,
                       _cl_command_run *command)
{
  /* Copy the currently set kernel arguments because the same kernel
     object can be reused for new launches with different arguments. */
  command->arguments = (struct pocl_argument *)malloc (
      (kernel->meta->num_args) * sizeof (struct pocl_argument));

  if (command->arguments == NULL && kernel->meta->num_args > 0)
    return CL_OUT_OF_HOST_MEMORY;

  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument *arg = &command->arguments[i];
      memcpy (arg, &src_arguments[i], sizeof (pocl_argument));

      if (arg->value != NULL)
        {
          size_t arg_alloc_size = arg->size;
          assert (arg_alloc_size > 0);
          /* FIXME: this is a kludge to determine an acceptable alignment,
           * we should probably extract the argument alignment from the
           * LLVM bytecode during kernel header generation. */
          size_t arg_alignment = pocl_size_ceil2 (arg_alloc_size);
          if (arg_alignment >= MAX_EXTENDED_ALIGNMENT)
            arg_alignment = MAX_EXTENDED_ALIGNMENT;
          if (arg_alloc_size < arg_alignment)
            arg_alloc_size = arg_alignment;

          arg->value = pocl_aligned_malloc (arg_alignment, arg_alloc_size);
          POCL_RETURN_ERROR_COND ((arg->value == NULL), CL_OUT_OF_HOST_MEMORY);
          memcpy (arg->value, src_arguments[i].value, arg->size);
        }
    }

  return CL_SUCCESS;
}

static int
process_command_ndrange_properties (
  const cl_command_properties_khr *properties)
{
  if (properties == NULL)
    return CL_SUCCESS;

  cl_uint num_properties = 0;
  const cl_command_properties_khr *key = NULL;
  for (key = properties; *key != 0; key += 2)
    num_properties += 1;
  POCL_RETURN_ERROR_ON ((num_properties == 0), CL_INVALID_VALUE,
                        "Properties != NULL, but zero properties in array\n");
  unsigned i = 0;
  for (key = properties; *key != 0; key += 2, ++i)
    {
      switch (*key)
        {
        case CL_MUTABLE_DISPATCH_ASSERTS_KHR:
        case CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR:
          POCL_RETURN_ERROR_ON (
            1, CL_INVALID_VALUE,
            "cl_khr_command_buffer_mutable_dispatch is not supported\n");
          break;
        default:
          POCL_RETURN_ERROR_ON (1, CL_INVALID_VALUE,
                                "Unknown property value in "
                                "cl_command_properties_khr\n");
        }
    }

  return CL_SUCCESS;
}

cl_int
pocl_record_ndrange_kernel (cl_command_buffer_khr command_buffer,
  cl_command_queue command_queue,
  const cl_command_properties_khr *properties,
  cl_kernel kernel,
  struct pocl_argument *src_arguments,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_items_in_wait_list,
  const cl_sync_point_khr *sync_point_wait_list,
  cl_sync_point_khr *sync_point_p)
{
  _cl_command_node *cmd = NULL;

  int errcode = process_command_ndrange_properties (properties);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  errcode = pocl_ndrange_kernel_common (
    command_buffer, command_queue, properties, kernel, src_arguments, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_items_in_wait_list, NULL, NULL, sync_point_wait_list, sync_point_p,
    &cmd);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument_info *ai
        = &cmd->command.run.kernel->meta->arg_info[i];
      struct pocl_argument *a = &cmd->command.run.kernel->dyn_arguments[i];
      if (ai->type == POCL_ARG_TYPE_SAMPLER)
        POname (clRetainSampler) (cmd->command.run.arguments[i].value);
    }

  errcode = pocl_command_record (command_buffer, cmd, sync_point_p);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (cmd);
  return errcode;
}

cl_int
pocl_ndrange_kernel_common (
  cl_command_buffer_khr command_buffer,
  cl_command_queue command_queue,
  const cl_command_properties_khr *properties,
  cl_kernel kernel,
  struct pocl_argument *src_arguments,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_items_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event_p,
  const cl_sync_point_khr *sync_point_wait_list,
  cl_sync_point_khr *sync_point_p,
  _cl_command_node **cmd)
{
  size_t offset[3] = { 0, 0, 0 };
  size_t num_groups[3] = { 0, 0, 0 };
  size_t local[3] = { 0, 0, 0 };

  int errcode = 0;

  size_t raw_ptr_count = 0;

  /* A linked list of memobjects implicit migration data. */
  pocl_buffer_migration_info *buf_migrations = NULL;

  assert (command_buffer == NULL
          || (event_wait_list == NULL && event_p == NULL));
  assert (command_buffer != NULL
          || (sync_point_wait_list == NULL && sync_point_p == NULL));

  if (command_queue != NULL && command_buffer == NULL)
    {
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                              CL_INVALID_COMMAND_QUEUE);
    }

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON (
      (command_queue->context != kernel->context), CL_INVALID_CONTEXT,
      "kernel and command_queue are not from the same context\n");

  cl_uint program_dev_i = CL_UINT_MAX;
  cl_device_id realdev = pocl_real_dev (command_queue->device);
  for (unsigned i = 0; i < kernel->program->num_devices; ++i)
    {
      if (kernel->program->devices[i] == realdev)
        program_dev_i = i;
    }
  assert (program_dev_i < CL_UINT_MAX);

  errcode = pocl_kernel_calc_wg_size (
      command_queue, kernel, program_dev_i, work_dim,
      global_work_offset, global_work_size,
      local_work_size, offset, local, num_groups);
  POCL_RETURN_ERROR_ON (errcode != CL_SUCCESS, errcode,
                        "Error calculating wg size\n");

  errcode = pocl_kernel_collect_mem_objs (command_queue, kernel, src_arguments,
                                          &buf_migrations);
  POCL_RETURN_ERROR_ON (errcode != CL_SUCCESS, errcode,
                        "Error collecting mem objects for kernel arguments\n");

  if (command_buffer == NULL)
    {
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_NDRANGE_KERNEL, event_p,
        num_items_in_wait_list, event_wait_list, buf_migrations);
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_NDRANGE_KERNEL,
        num_items_in_wait_list, sync_point_wait_list, buf_migrations);
    }

  POCL_RETURN_ERROR_ON (errcode != CL_SUCCESS, errcode,
                        "Error constructing command struct\n");

  _cl_command_node *c = *cmd;
  c->program_device_i = program_dev_i;
  c->next = NULL;

  c->command.run.kernel = kernel;
  c->command.run.hash = kernel->meta->build_hash[program_dev_i];
  c->command.run.pc.local_size[0] = local[0];
  c->command.run.pc.local_size[1] = local[1];
  c->command.run.pc.local_size[2] = local[2];
  c->command.run.pc.work_dim = work_dim;
  c->command.run.pc.num_groups[0] = num_groups[0];
  c->command.run.pc.num_groups[1] = num_groups[1];
  c->command.run.pc.num_groups[2] = num_groups[2];
  c->command.run.pc.global_offset[0] = offset[0];
  c->command.run.pc.global_offset[1] = offset[1];
  c->command.run.pc.global_offset[2] = offset[2];

  errcode = POname (clRetainKernel) (kernel);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  errcode
    = pocl_kernel_copy_args (kernel, src_arguments, &(*cmd)->command.run);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_ndrange_node_cleanup (*cmd);
  pocl_mem_manager_free_command (*cmd);
  return errcode;
}
