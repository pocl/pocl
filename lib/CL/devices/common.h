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
#include "utlist.h"

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

POCL_EXPORT
void pocl_fill_dev_image_t (dev_image_t *di, struct pocl_argument *parg,
                            cl_device_id device);

POCL_EXPORT
void pocl_fill_dev_sampler_t (dev_sampler_t *ds, struct pocl_argument *parg);

void pocl_copy_mem_object (cl_device_id dest_dev, cl_mem dest,
                           size_t dest_offset,
                           cl_device_id source_dev, cl_mem source,
                           size_t source_offset, size_t cb);

void pocl_migrate_mem_objects (_cl_command_node *node);

void pocl_scheduler (_cl_command_node * volatile * ready_list,
                     pthread_mutex_t *lock_ptr);

POCL_EXPORT
void pocl_exec_command (_cl_command_node * volatile node);

POCL_EXPORT
void pocl_ndrange_node_cleanup(_cl_command_node *node);
void pocl_mem_objs_cleanup (cl_event event);

POCL_EXPORT
void pocl_broadcast (cl_event event);

POCL_EXPORT
void pocl_init_dlhandle_cache ();

char *pocl_check_kernel_disk_cache (_cl_command_node *cmd, int specialized);

size_t pocl_cmd_max_grid_dim_width (_cl_command_run *cmd);

POCL_EXPORT
void pocl_check_kernel_dlhandle_cache (_cl_command_node *command,
                                       unsigned initial_refcount,
                                       int specialize);

POCL_EXPORT
void pocl_release_dlhandle_cache (_cl_command_node *cmd);

void pocl_setup_device_for_system_memory(cl_device_id device);

void pocl_reinit_system_memory();

POCL_EXPORT
void pocl_set_buffer_image_limits(cl_device_id device);

POCL_EXPORT
void* pocl_aligned_malloc_global_mem(cl_device_id device, size_t align, size_t size);

POCL_EXPORT
void pocl_free_global_mem(cl_device_id device, void *ptr, size_t size);

void pocl_print_system_memory_stats();

void pocl_calculate_kernel_hash (cl_program program, unsigned kernel_i,
                                 unsigned device_i);

POCL_EXPORT
void pocl_init_default_device_infos (cl_device_id dev);

#ifdef __cplusplus
}
#endif

#endif
