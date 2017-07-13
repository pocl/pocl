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

/* Euclid's algorithm for the Greatest Common Divisor */
static inline size_t
gcd (size_t a, size_t b)
{
  int c;
  while (a) {
    c = a; a = b % a; b = c;
  }
  return b;
}

/* Find the largest divisor of dividend which is less than limit */
static inline size_t
upper_divisor (size_t dividend, size_t limit)
{
  /* The algorithm is currently not very smart, we
   * start from limit and subtract until we find something
   * that divides dividend. In optimal conditions this is found
   * quickly, but it takes limit steps if dividend is prime.
   * TODO FIXME improve algorithm
   */
  if (dividend < limit) return dividend; // small optimization
  assert (limit > 0); // should never be called with limit == 0
  while (dividend % limit != 0) --limit;
  return limit;
}

/* Check that a divides b and b divides c */
static inline int
divide_chain (size_t a, size_t b, size_t c)
{
  return (b % a == 0 && c % b == 0);
}


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

  int b_migrate_count, buffer_count;
  unsigned i;
  int errcode = 0;
  cl_device_id realdev = NULL;
  struct pocl_context pc;
  _cl_command_node *command_node;
  /* alloc from stack to avoid malloc. num_args is the absolute max needed */
  cl_mem mem_list[kernel->num_args + 1];
  /* reserve space for potential buffer migrate events */
  cl_event new_event_wait_list[num_events_in_wait_list + kernel->num_args + 1];

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);

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

  for (i = 0; i < kernel->num_args; i++)
    {
      POCL_RETURN_ERROR_ON((!kernel->arg_info[i].is_set),
        CL_INVALID_KERNEL_ARGS, "The %i-th kernel argument is not set!\n", i);
    }

  max_local_x = command_queue->device->max_work_item_sizes[0];
  max_local_y = command_queue->device->max_work_item_sizes[1];
  max_local_z = command_queue->device->max_work_item_sizes[2];
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
  /* otherwise, if the local work size was not specified find the optimal one.
   * Note that at some point we also checked for local > global. This doesn't
   * make sense while we only have 1.2 support for kernel enqueue (and
   * when only uniform group sizes are allowed), but it might turn useful
   * when picking the hardware sub-group size in more sophisticated
   * 2.0 support scenarios.
   */
  else if (local_work_size == NULL)
    {
      /* Embarrassingly parallel kernel with a free work-group size. Try to
       * figure out one which utilizes all the resources efficiently. Assume
       * work-groups are scheduled to compute units, so try to split it to a
       * number of work groups at the equal to the number of CUs, while still
       * trying to respect the preferred WG size multiple (for better SIMD
       * instruction utilization).
      */
      size_t preferred_wg_multiple; cl_int prop_err =
        POname(clGetKernelWorkGroupInfo) (kernel, command_queue->device,
          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof (size_t),
          &preferred_wg_multiple, NULL);

      if (prop_err != CL_SUCCESS) /* unlikely */
        preferred_wg_multiple = 1;

      POCL_MSG_PRINT_INFO("Preferred WG size multiple %zu\n",
                          preferred_wg_multiple);

      /* However, we have some constraints about the local size:
       * 1. local_{x,y,z} must divide global_{x,y,z} exactly, at least
       *    as long as we only support uniform group sizes (i.e. OpenCL 1.x);
       * 2. each of local_{x,y,z} must be less than the corresponding max size
       *    for the device;
       * 3. the product of local_{x,y,z} must be less than the maximum local
       *    work-group size.
       *
       * Due to constraint 1., we may not have the possibility to proceed by
       * multiples of the preferred_wg_multiple (e.g. if preferred = 16 and
       * global size = 24). Our stepping granularity in each direction will
       * therefore be the GCD of the global size in that direction and the
       * preferred wg size.
       *
       * Note that the grain might actually be as low as 1, if the two values
       * are coprimes (e.g. preferred = 8, global size = 17). There is no good
       * solution in this case, and there's nothing we can do about it. On the
       * opposite side of the spectrum, we might be lucky and grain_* =
       * preferred_wg_multiple (this is the case e.g. if the programmer already
       * checked for the preferred wg multiple and rounded the global size up
       * to the multiple of it).
       */

      const size_t grain_x = gcd (preferred_wg_multiple, global_x);
      const size_t grain_y = gcd (preferred_wg_multiple, global_y);
      const size_t grain_z = gcd (preferred_wg_multiple, global_z);

      /* We now want to get the largest multiple of the grain size that still
       * divides global_* _and_ is less than the maximum local size in each
       * direction.
       *
       * So we have G = K*g and we want to find k such that k*g < M and
       * k*g still divides G, i.e. k must divide K.
       * The largest multiple of g that is less than M can be found as
       * (M/g)*g (integer division), so our upper bound for k is k' = M/g.
       */

      /*                      /------- K ------\  /-------- k' -------\  */
      local_x = upper_divisor (global_x / grain_x, max_local_x / grain_x);
      local_y = upper_divisor (global_y / grain_y, max_local_y / grain_y);
      local_z = upper_divisor (global_z / grain_z, max_local_z / grain_z);

      local_x *= grain_x;
      local_y *= grain_y;
      local_z *= grain_z;

      /* So we now have the largest possible local sizes that divide the global
       * sizes while being multiples of the grain size.
       * We still have to ensure that the work-group size overall is not larger
       * than the maximum allowed, and we have to do this while preserving the
       * 'local divides global' condition, and we would like to preserve the
       * 'multiple of grain' too, if possible.
       * We always reduce z first, then y, then x, on the assumption that
       * kernels will work with x varying faster, and thus being a better
       * vectorization candidate, followed by y and then by z. (This assumption
       * is in some sense sanctioned by the standard itself, see e.g. the
       * get_{global,local}_linear_id functions in OpenCL 2.x)
       * TODO this might not be optimal in all cases. For example, devices with
       * a hardware sampler might benefit from more evenly sized work-groups
       * for kernels that use images. Some kind of kernel + device analysis
       * would be needed here.
       */

      while (local_x * local_y * local_z > max_group_size)
        {
          /* We are going to try three strategies, in order:
           *
           * Halving a coordinate, if the halved coordinate is still a multiple
           * of the grain size and a divisor of the global size.
           *
           * Setting the coordinates with the smallest grain to 1,
           * since they aren't good candidates for vectorizations anyway.
           *
           * Setting to 1 any coordinate, as a desperate measure.
           */

#define TRY_HALVE(coord) \
if ((local_##coord & 1) == 0 && \
    divide_chain (grain_##coord, local_##coord/2, global_##coord)) \
  { \
    local_##coord /= 2; \
    continue; \
  }

#define TRY_LEAST_GRAIN(c1, c2, c3) \
if (local_##c1 > 1 && grain_##c1 <= grain_##c2 && grain_##c1 <= grain_##c3) \
  { \
    local_##c1 = 1; \
    continue; \
  }

#define DESPERATE_CASE(coord) \
if (local_##coord > 1) \
  { \
    local_##coord = 1; \
    continue; \
  }
          /* Halving attempt first */
          TRY_HALVE(z) else TRY_HALVE(y) else TRY_HALVE(x)

          /* Ok no luck. Find the coordinate with the smallest grain and
           * kill that */
          TRY_LEAST_GRAIN(z, x, y) else
          TRY_LEAST_GRAIN(y, z, x) else
          TRY_LEAST_GRAIN(x, y, z)

          /* No luck either? Give up, kill everything */
          DESPERATE_CASE(z) else DESPERATE_CASE(y) else DESPERATE_CASE(x)
#undef DESPERATE_CASE
#undef TRY_LEAST_GRAIN
#undef TRY_HALVE
        }

      /* We now have the largest possible local work-group size that satisfies
       * all the hard constraints (divide global, per-dimension bound, overall
       * bound) and our soft constraint of being as close as possible a
       * multiple of the preferred work-group size multiple. Such a greedy
       * algorithm minimizes the total number of work-groups. In moderate-sized
       * launch grid, this may result in less work-groups than the number of
       * Compute Units, with a resulting imbalance in the workload
       * distribution. At the same time, we want to avoid too many work-groups,
       * since some devices are penalized by such fragmentation. Finding a good
       * balance between the two is a hard problem, and generally depends on
       * the device as well as the kernel utilization of its resources.
       * Lacking that, as a first step we will simply try to guarantee that we
       * have at least one work-group per CU, as long as the local work size
       * does not drop below a given threshold.
       */

      /* Pick a minimum work-group size of 4 times the preferred work-group
       * size multiple, under the assumption that this would be a good
       * candidate below which a Compute Unit will not do enough work.
       */
      const size_t min_group_size = 4 * preferred_wg_multiple;

      /* We need the number of Compute Units in the device, since we want
       * at least that many work-groups, if possible */

      cl_uint ncus = command_queue->device->max_compute_units;

      /* number of workgroups */
      size_t nwg_x = global_x / local_x;
      size_t nwg_y = global_y / local_y;
      size_t nwg_z = global_z / local_z;

      size_t splits; /* number of splits to bring ngws to reach ncu */
      /* Only proceed if splitting wouldn't bring us below the minimum
       * group size */
      while (((splits = ncus / (nwg_x * nwg_y * nwg_z)) > 1) &&
             (local_x * local_y * local_z > splits * min_group_size))
        {
          /* Very simple splitting approach: find a dimension divisible by
           * split, and lacking that divide by something less, if possible.
           * If we fail at splitting at all, we will try killing the smaller of
           * the dimensions.
           * We will set splits to 0 if we succeed in the TRY_SPLIT, so that
           * we know that we can skip the rest.
           * If we get to the end of the while without splitting and without
           * killing a dimension, we bail out early because it means we
           * couldn't do anything useful without dropping below min_group_size.
           */

#define TRY_SPLIT(coord) \
if ((local_##coord % splits) == 0 && \
    divide_chain (grain_##coord, local_##coord/splits, global_##coord)) \
  { \
    local_##coord /= splits; nwg_##coord *= splits; splits = 0; \
    continue; \
  }

#define TRY_LEAST_DIM(c1, c2, c3) \
if (local_##c1 > 1 && local_##c1 <= local_##c2 && local_##c1 <= local_##c3 && \
    local_##c2*local_##c3 >= min_group_size) \
  { \
    local_##c1 = 1; nwg_##c1 = global_##c1; \
    continue; \
  }

          while (splits > 1)
            {
              TRY_SPLIT(z) else TRY_SPLIT(y) else TRY_SPLIT(x)
                else splits--;
            }
          /* When we get here, splits will be 0 if we split, 1 if we failed:
           * in which case we will just kill one of the dimensions instead,
           * using the same TRY_LEAST_GRAIN and DESPERATE_CASE seen before
           */
          if (splits == 0)
            continue;

          TRY_LEAST_DIM(z, x, y) else TRY_LEAST_DIM(y, z, x) else
          TRY_LEAST_DIM(x, y, z) else break;
#undef TRY_LEAST_DIM
#undef TRY_SPLIT
        }
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

  char cachedir[POCL_FILENAME_LENGTH];
  int realdev_i = pocl_cl_device_to_index (kernel->program, realdev);
  assert (realdev_i >= 0);
  pocl_cache_kernel_cachedir_path (cachedir, kernel->program,
                                   realdev_i, kernel, "",
                                   local_x, local_y, local_z);

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

  errcode = pocl_create_command (&command_node, command_queue,
                               CL_COMMAND_NDRANGE_KERNEL, event,
                               num_events_in_wait_list + b_migrate_count,
                               (num_events_in_wait_list + b_migrate_count)?
                               new_event_wait_list : NULL,
                               buffer_count, mem_list);
  if (errcode != CL_SUCCESS)
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
  errcode = CL_SUCCESS;

ERROR:
  return errcode;

}
POsym(clEnqueueNDRangeKernel)
