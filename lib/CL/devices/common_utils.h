/* common_utils.h - common utilities for CPU device drivers

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and
                 2012-2018 Pekka Jääskeläinen / Tampere Univ. of Technology and
                 2021 Tobias Baumann / Zuse Institute Berlin

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

#ifndef POCL_PTHREAD_UTILS_H
#define POCL_PTHREAD_UTILS_H

#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_context.h"
#include "pocl_workgroup_func.h"
#ifdef HAVE_LIBJPEG_TURBO
#include "cpu_dbk/pocl_dbk_khr_jpeg_cpu.h"
#endif
#ifdef HAVE_ONNXRT
#include "cpu_dbk/pocl_dbk_khr_onnxrt_cpu.h"
#endif

/* Generic struct for CPU device drivers.
 * Not all fields of this struct are used by all drivers. */
typedef struct kernel_run_command kernel_run_command;
struct kernel_run_command
{
  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) pocl_lock_t lock;
  void *data;
  cl_kernel kernel;
  cl_device_id device;
  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) struct pocl_context pc;
  _cl_command_node *cmd;
  pocl_workgroup_func workgroup;
  struct pocl_argument *kernel_args;
  kernel_run_command *prev;
  kernel_run_command *next;
  unsigned long ref_count;

  /* actual kernel arguments. these are setup once at the kernel setup
   * phase, then each thread sets up the local arguments for itself. */
  void **arguments;
  /* this is required b/c there's an additional level of indirection */
  void **arguments2;

  size_t remaining_wgs;
  size_t wgs_dealt;
};

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef USE_POCL_MEMMANAGER
void pocl_init_kernel_run_command_manager ();
void pocl_init_thread_argument_manager ();
kernel_run_command* new_kernel_run_command ();
void free_kernel_run_command (kernel_run_command *k);
#else
#define pocl_init_kernel_run_command_manager() NULL
#define pocl_init_thread_argument_manager() NULL
#define new_kernel_run_command()                                              \
  (kernel_run_command *)pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE,         \
                                             sizeof (kernel_run_command))
#define free_kernel_run_command(k) pocl_aligned_free (k)
#endif

POCL_EXPORT
cl_int pocl_cpu_init_common (cl_device_id device);

POCL_EXPORT
int pocl_cpu_supports_dbk (cl_device_id device,
                           BuiltinKernelId kernel_id,
                           const void *kernel_attributes);
POCL_EXPORT
int pocl_cpu_build_defined_builtin (cl_program program, cl_uint device_i);

POCL_EXPORT
int pocl_cpu_execute_dbk (cl_program program,
                          cl_kernel kernel,
                          pocl_kernel_metadata_t *meta,
                          cl_uint dev_i,
                          struct pocl_argument *arguments);

POCL_EXPORT
void pocl_cpu_probe ();

POCL_EXPORT
void pocl_setup_kernel_arg_array (kernel_run_command *k);

POCL_EXPORT
int pocl_setup_kernel_arg_array_with_locals (void **arguments,
                                             void **arguments2,
                                             kernel_run_command *k,
                                             char *local_mem,
                                             size_t local_mem_size);

POCL_EXPORT
int pocl_tensor_type_size (cl_tensor_datatype T);

POCL_EXPORT
int pocl_tensor_type_is_int (cl_tensor_datatype T);

POCL_EXPORT
void pocl_free_kernel_arg_array (kernel_run_command *k);

POCL_EXPORT
void pocl_free_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                        kernel_run_command *k);

void *pocl_cpu_get_ptr (struct pocl_argument *arg, unsigned global_mem_id);

#ifdef __cplusplus
}
#endif

#endif /* COMMON_UTILS_H */
