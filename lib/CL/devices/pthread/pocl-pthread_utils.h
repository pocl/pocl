#ifndef POCL_PTHREAD_UTILS_H
#define POCL_PTHREAD_UTILS_H

#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* locking macros */
#define PTHREAD_LOCK(__lock)  pthread_mutex_lock(__lock)
#define PTHREAD_UNLOCK(__lock) pthread_mutex_unlock(__lock)
#define PTHREAD_INIT_LOCK(__lock) pthread_mutex_init(__lock, NULL)
#define PTHREAD_DESTROY_LOCK(__lock) pthread_mutex_destroy(__lock)

/* Apparently Mac OS X does not have spinlock, despite having pthreads.
 * for now only enable spinlocks on linux.*/
#ifdef __linux__
  #define PTHREAD_FAST_LOCK_T pthread_spinlock_t
  #define PTHREAD_FAST_LOCK(l) pthread_spin_lock(l)
  #define PTHREAD_FAST_UNLOCK(l) pthread_spin_unlock(l)
  #define PTHREAD_FAST_INIT(l) pthread_spin_init(l, PTHREAD_PROCESS_PRIVATE)
  #define PTHREAD_FAST_DESTROY(l) pthread_spin_destroy(l)
#else
  #define PTHREAD_FAST_LOCK_T pthread_mutex_t
  #define PTHREAD_FAST_LOCK(l) pthread_mutex_lock(l)
  #define PTHREAD_FAST_UNLOCK(l) pthread_mutex_unlock(l)
  #define PTHREAD_FAST_INIT(l) pthread_mutex_init(l, NULL)
  #define PTHREAD_FAST_DESTROY(l) pthread_mutex_destroy(l)
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

  PTHREAD_FAST_LOCK_T lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

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
