/* common.h - common code that can be reused between device driver
              implementations

   Copyright (c) 2012-2019 Pekka Jääskeläinen

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

#ifndef POCL_COMMON_H
#define POCL_COMMON_H

#include "pocl_cl.h"

#define __CBUILD__
#include "pocl_image_types.h"
#undef __CBUILD__

#define XSETUP_DEVICE_CL_VERSION(A, B)             \
  dev->cl_version_int = (A * 100) + (B * 10);     \
  dev->cl_version_std = "CL" # A "." # B;         \
  dev->version = "OpenCL " # A "." # B " pocl";

#define SETUP_DEVICE_CL_VERSION(a, b) XSETUP_DEVICE_CL_VERSION(a, b)

#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR    1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT   1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT     1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG    1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT   1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE  1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR       1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT      1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT        1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG       1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT      1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE     1

/* Half is internally represented as short */
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_HALF POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT

#ifdef __cplusplus  
extern "C" {
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

void pocl_init_cpu_device_infos (cl_device_id dev);

void pocl_init_cpu_global_mem (cl_device_id dev);

void pocl_bufalloc_init_global_mem (cl_device_id device, size_t size,
                                    void *data, void *data2);
void pocl_global_mem_print_stats (pocl_global_mem_t *gmem);
int pocl_global_mem_can_allocate (pocl_global_mem_t *gmem,
                                  pocl_mem_identifier *p);
void pocl_global_mem_allocated (pocl_global_mem_t *gmem,
                                pocl_mem_identifier *p);
void pocl_global_mem_freed (pocl_global_mem_t *gmem, pocl_mem_identifier *p);
uint64_t pocl_driver_memobj_device_size (cl_device_id dev,
                                         uint64_t input_size);

void fill_dev_image_t (dev_image_t *di, struct pocl_argument *parg,
                       cl_device_id device);

void fill_dev_sampler_t (dev_sampler_t *ds, struct pocl_argument *parg);

void pocl_copy_mem_object (cl_device_id dest_dev, cl_mem dest,
                           size_t dest_offset,
                           cl_device_id source_dev, cl_mem source,
                           size_t source_offset, size_t cb);

void pocl_migrate_mem_objects (_cl_command_node *node);

void pocl_scheduler (_cl_command_node * volatile * ready_list,
                     pthread_mutex_t *lock_ptr);

void pocl_exec_command (_cl_command_node * volatile node);

void pocl_ndrange_node_cleanup(_cl_command_node *node);
void pocl_mem_objs_cleanup (cl_event event);

void pocl_broadcast (cl_event event);

void pocl_init_dlhandle_cache ();

char *pocl_check_kernel_disk_cache (_cl_command_node *cmd, int specialize);

size_t pocl_cmd_max_grid_dim_width (struct pocl_context *pc);

void generate_spec_suffix (char *suffix, int specialized,
                           struct pocl_context *pc, cl_device_id dev);

void pocl_check_kernel_dlhandle_cache (_cl_command_node *command,
                                       unsigned initial_refcount,
                                       int specialize);

void pocl_release_dlhandle_cache (_cl_command_node *cmd);

void pocl_set_buffer_image_limits(cl_device_id device);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
