/* common_utils.h - common utilities for pthread and tbb devices

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and
                 2012-2018 Pekka Jääskeläinen / Tampere Univ. of Technology and
                 2021 Tobias Baumann / Zuse Institute Berlin

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

#ifndef POCL_PTHREAD_UTILS_H
#define POCL_PTHREAD_UTILS_H

#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_context.h"
#include "pocl_workgroup_func.h"

/* NOTE: Some entries in this struct are only required by the pthread device and
 * not by the tbb device. However, they have been kept for the tbb device to
 * simplify the code structure and the maintenance effort. */
typedef struct kernel_run_command kernel_run_command;
struct kernel_run_command
{
  void *data;
  cl_kernel kernel;
  cl_device_id device;
  _cl_command_node *cmd;
  pocl_workgroup_func workgroup;
  struct pocl_argument *kernel_args;
  kernel_run_command *prev;
  kernel_run_command *next;
  unsigned long ref_count; /* pthread device only */

  /* actual kernel arguments. these are setup once at the kernel setup
   * phase, then each thread sets up the local arguments for itself. */
  void **arguments;
  /* this is required b/c there's an additional level of indirection */
  void **arguments2;

  /* pthread device only */
  POCL_FAST_LOCK_T lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  /* both only used by the pthread device */
  size_t remaining_wgs __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  size_t wgs_dealt;

  struct pocl_context pc __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

} __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

struct data {
    /* Currently loaded kernel. */
    cl_kernel current_kernel;
    volatile uint64_t total_cmd_exec_time;
};

#ifdef __cplusplus
extern "C" {
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
#define free_kernel_run_command(k) free (k)
#endif

POCL_EXPORT
cl_int pocl_device_init_common (cl_device_id device);

POCL_EXPORT
void setup_kernel_arg_array (kernel_run_command *k);

POCL_EXPORT
void setup_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                         kernel_run_command *k,
                                         char *local_mem,
                                         size_t local_mem_size);

POCL_EXPORT
void free_kernel_arg_array (kernel_run_command *k);

POCL_EXPORT
void free_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                        kernel_run_command *k);

#ifdef __cplusplus
}
#endif

#endif /* COMMON_UTILS_H */
