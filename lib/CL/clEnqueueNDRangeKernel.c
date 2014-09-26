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
#include "utlist.h"
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
  char tmpdir[POCL_FILENAME_LENGTH];
  char kernel_filename[POCL_FILENAME_LENGTH];
  char parallel_filename[POCL_FILENAME_LENGTH];
  char so_filename[POCL_FILENAME_LENGTH];
  int i, count;
  int error;
  struct pocl_context pc;
  _cl_command_node *command_node;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);
  
  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON((command_queue->context != kernel->context),
    CL_INVALID_CONTEXT, "kernel and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_COND((work_dim < 1), CL_INVALID_WORK_DIMENSION);
  POCL_RETURN_ERROR_ON((work_dim > command_queue->device->max_work_item_dimensions),
    CL_INVALID_WORK_DIMENSION, "work_dim exceeds devices' max workitem dimensions\n");

  assert(command_queue->device->max_work_item_dimensions <= 3);

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
        "The %i-th kernel argument is not set!\n", i)
    }

  if (local_work_size != NULL) 
    {
      local_x = local_work_size[0];
      local_y = work_dim > 1 ? local_work_size[1] : 1;
      local_z = work_dim > 2 ? local_work_size[2] : 1;
    } 
  else 
    {
      size_t preferred_wg_multiple;
      cl_int retval = 
        POname(clGetKernelWorkGroupInfo)
        (kernel, command_queue->device, 
         CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
         sizeof (size_t), &preferred_wg_multiple, NULL);

      local_x = local_y = local_z = 1;
      if (retval == CL_SUCCESS)
        {
          /* Find the largest multiple of the preferred wg multiple.
             E.g. if the preferred is 8 it doesn't work with a
             global size of 20. However, 4 is better than 1 in that
             case because it still enables wi-parallelization. */
          while (preferred_wg_multiple >= 1)
            {
              if (global_x % preferred_wg_multiple == 0 &&
                  preferred_wg_multiple <= global_x)
                {
                  local_x = preferred_wg_multiple;
                  break;
                }
              preferred_wg_multiple /= 2;
            }
        }
    }

#ifdef DEBUG_NDRANGE
  printf("### queueing kernel %s for dimensions %zu x %zu x %zu...", 
         kernel->function_name, local_x, local_y, local_z);
#endif

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


  snprintf (tmpdir, POCL_FILENAME_LENGTH, "%s/%s/%s/%zu-%zu-%zu", 
            kernel->program->temp_dir, command_queue->device->short_name, 
            kernel->name, 
            local_x, local_y, local_z);

  if (access (tmpdir, F_OK) != 0)
    mkdir (tmpdir, S_IRWXU);
  
  error = snprintf
          (parallel_filename, POCL_FILENAME_LENGTH,
          "%s/%s", tmpdir, POCL_PARALLEL_BC_FILENAME);
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;

  error = snprintf
          (so_filename, POCL_FILENAME_LENGTH,
          "%s/%s.so", tmpdir, kernel->name);
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;

  error = snprintf
          (kernel_filename, POCL_FILENAME_LENGTH,
           "%s/%s/%s", kernel->program->temp_dir,
           command_queue->device->short_name, POCL_PROGRAM_BC_FILENAME);
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;

  if (access(so_filename, F_OK) != 0)
    {
      error = pocl_llvm_generate_workgroup_function
          (command_queue->device,
           kernel, local_x, local_y, local_z,
           parallel_filename, kernel_filename);

      if (error)  return error;
    }

  error = pocl_create_command (&command_node, command_queue,
                               CL_COMMAND_NDRANGE_KERNEL,
                               event, num_events_in_wait_list,
                               event_wait_list);
  if (error != CL_SUCCESS)
    return error;

  pc.work_dim = work_dim;
  pc.num_groups[0] = global_x / local_x;
  pc.num_groups[1] = global_y / local_y;
  pc.num_groups[2] = global_z / local_z;
  pc.global_offset[0] = offset_x;
  pc.global_offset[1] = offset_y;
  pc.global_offset[2] = offset_z;

  command_node->type = CL_COMMAND_NDRANGE_KERNEL;
  command_node->command.run.data = command_queue->device->data;
  command_node->command.run.tmp_dir = strdup(tmpdir);
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

  command_node->command.run.arg_buffer_count = 0;
  for (i = 0; i < kernel->num_args; ++i)
  {
    struct pocl_argument *al = &(kernel->dyn_arguments[i]);
    if (!kernel->arg_info[i].is_local && kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER && al->value != NULL)
      ++command_node->command.run.arg_buffer_count;
  }
  
  /* Copy the argument buffers just so we can free them after execution. */
  command_node->command.run.arg_buffers = 
    (cl_mem *) malloc (sizeof (struct _cl_mem) * command_node->command.run.arg_buffer_count);
  count = 0;
  for (i = 0; i < kernel->num_args; ++i)
  {
    struct pocl_argument *al = &(kernel->dyn_arguments[i]);
    if (!kernel->arg_info[i].is_local && kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER && al->value != NULL)
      {
        cl_mem buf;
#if 0
        printf ("### retaining arg %d - the buffer %x of kernel %s\n", i, buf, kernel->function_name);
#endif
        buf = *(cl_mem *) (al->value);
        /* Retain all memobjects so they won't get freed before the
           queued kernel has been executed. */
        if (buf != NULL)
          POname(clRetainMemObject) (buf);
        command_node->command.run.arg_buffers[count] = buf;
        ++count;
      }
  }

  pocl_command_enqueue (command_queue, command_node);

  return CL_SUCCESS;
}
POsym(clEnqueueNDRangeKernel)
