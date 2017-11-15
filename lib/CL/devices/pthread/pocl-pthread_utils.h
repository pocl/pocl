#ifndef POCL_PTHREAD_UTILS_H
#define POCL_PTHREAD_UTILS_H

#include "pocl_cl.h"

/* locking macros */
#define PTHREAD_LOCK(__lock)                                                  \
  do                                                                          \
    {                                                                         \
      pthread_mutex_lock ((__lock));                                          \
    }                                                                         \
  while (0)

#define PTHREAD_UNLOCK(__lock)                                                \
  do                                                                          \
    {                                                                         \
      pthread_mutex_unlock ((__lock));                                        \
    }                                                                         \
  while (0)

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
  kernel_run_command *volatile next;
  volatile int ref_count;

#ifdef POCL_PTHREAD_CACHE_MONITORING
  pocl_cache_data cache_data;
#endif

  pthread_mutex_t lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  volatile unsigned remaining_wgs __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  volatile unsigned wgs_dealt;

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
#define new_kernel_run_command() \
  (kernel_run_command*) calloc (1, sizeof(kernel_run_command))
#define free_kernel_run_command(k) free (k)
#endif
void setup_kernel_arg_array(void **arguments, kernel_run_command *k);
void free_kernel_arg_array (void **arguments, kernel_run_command *k);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
