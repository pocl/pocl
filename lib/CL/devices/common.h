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

#include "pocl.h"
#include "utlist.h"

#define __CBUILD__
#include "pocl_image_types.h"
#undef __CBUILD__

#define XSETUP_DEVICE_CL_VERSION(D, A, B)                                     \
  D->version_as_int = (A * 100) + (B * 10);                                   \
  D->version_as_cl = CL_MAKE_VERSION (A, B, 0);                               \
  D->version = "OpenCL " #A "." #B " PoCL";                                   \
  D->opencl_c_version_as_opt = "CL" #A "." #B;                                \
  D->opencl_c_version_as_cl = CL_MAKE_VERSION (A, B, 0);

#define SETUP_DEVICE_CL_VERSION(D, a, b) XSETUP_DEVICE_CL_VERSION (D, a, b)

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

POCL_EXPORT
void pocl_exec_command (_cl_command_node *node);

POCL_EXPORT
char *pocl_cpu_build_hash (cl_device_id device);

POCL_EXPORT
void pocl_broadcast (cl_event event);

POCL_EXPORT
void pocl_init_dlhandle_cache ();

POCL_EXPORT
int pocl_check_kernel_disk_cache (char *module_fn,
                                  _cl_command_node *cmd,
                                  int specialized);

POCL_EXPORT
size_t pocl_cmd_max_grid_dim_width (_cl_command_run *cmd);

POCL_EXPORT
void *pocl_check_kernel_dlhandle_cache (_cl_command_node *command,
                                        int retain,
                                        int specialize);

POCL_EXPORT
void pocl_release_dlhandle_cache (void *dlhandle_cache_item);

POCL_EXPORT
void pocl_setup_device_for_system_memory(cl_device_id device);

POCL_EXPORT
void pocl_reinit_system_memory();

POCL_EXPORT
void pocl_set_buffer_image_limits(cl_device_id device);

void pocl_print_system_memory_stats();

POCL_EXPORT
void pocl_init_default_device_infos (cl_device_id dev,
                                     const char *device_extensions);

POCL_EXPORT
void pocl_setup_opencl_c_with_version (cl_device_id dev, int supports_30);

POCL_EXPORT
void pocl_setup_extensions_with_version (cl_device_id dev);

POCL_EXPORT
void pocl_setup_ils_with_version (cl_device_id dev);

POCL_EXPORT
void pocl_setup_features_with_version (cl_device_id dev);

POCL_EXPORT
void pocl_setup_builtin_kernels_with_version (cl_device_id dev);

POCL_EXPORT
void __printf_flush_buffer (char *buffer, uint32_t buffer_size);

POCL_EXPORT
void pocl_write_printf_buffer (char *printf_buffer, uint32_t bytes);

#ifdef __cplusplus
}
#endif

#endif
