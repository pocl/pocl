#ifndef POCL_PTHREAD_UTILS_H
#define POCL_PTHREAD_UTILS_H

#include "pocl_cl.h"
#include "pocl_util.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

typedef struct kernel_run_command kernel_run_command;
struct kernel_run_command
{
  void *data;
  cl_kernel kernel;
  cl_device_id device;
  _cl_command_node *cmd;
  pocl_workgroup workgroup;
  struct pocl_argument *kernel_args;
  kernel_run_command *next;
  unsigned long ref_count;

  /* actual kernel arguments. these are setup once at the kernel setup
   * phase, then each thread sets up the local arguments for itself. */
  void **arguments;
  /* this is required b/c there's an additional level of indirection */
  void **arguments2;

#ifdef POCL_PTHREAD_CACHE_MONITORING
  pocl_cache_data cache_data;
#endif

  POCL_FAST_LOCK_T lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  unsigned remaining_wgs __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  unsigned wgs_dealt;

  struct pocl_context pc __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

} __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

#ifdef USE_POCL_MEMMANAGER
void pocl_init_kernel_run_command_manager (void);
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

void setup_kernel_arg_array (kernel_run_command *k);
void setup_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                         kernel_run_command *k,
                                         char *local_mem,
                                         size_t local_mem_size);
void free_kernel_arg_array (kernel_run_command *k);
void free_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                        kernel_run_command *k);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
