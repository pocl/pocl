
#include <string.h>
#include "pocl-pthread_utils.h"
#include "utlist.h"
#include "common.h"
#include "pocl-pthread.h"
#include "pocl_mem_management.h"

#ifdef USE_POCL_MEMMANAGER

static kernel_run_command *volatile kernel_pool = 0;
static int kernel_pool_initialized = 0;
static pocl_lock_t kernel_pool_lock;


void pocl_init_kernel_run_command_manager (void)
{
  if (!kernel_pool_initialized)
    {
      kernel_pool_initialized = 1;
      POCL_INIT_LOCK (kernel_pool_lock);
    }
}

void pocl_init_thread_argument_manager (void)
{
  if (!kernel_pool_initialized)
    {
      kernel_pool_initialized = 1;
      POCL_INIT_LOCK (kernel_pool_lock);
    }
}

kernel_run_command* new_kernel_run_command ()
{
  kernel_run_command *volatile k = NULL;
  POCL_LOCK (kernel_pool_lock);
  if ((k = kernel_pool))
    {
      LL_DELETE (kernel_pool, k);
      memset (k, 0, sizeof(kernel_run_command));
      pthread_mutex_init(&k->lock, NULL);
      POCL_UNLOCK (kernel_pool_lock);
      return k;
    }

  POCL_UNLOCK (kernel_pool_lock);
  k = (kernel_run_command*)calloc (1, sizeof (kernel_run_command));
  pthread_mutex_init (&k->lock, NULL);
  return k;
}

void free_kernel_run_command (kernel_run_command *k)
{
  POCL_LOCK (kernel_pool_lock);
  pthread_mutex_destroy (&k->lock);
  LL_PREPEND (kernel_pool, k);
  POCL_UNLOCK (kernel_pool_lock);
}

#endif

#define ARGS_SIZE                                                             \
  (sizeof (void *) * (kernel->num_args + kernel->num_locals + 1))

static char *
align_ptr (char *p)
{
  uintptr_t r = (uintptr_t)p;
  if (r & (MAX_EXTENDED_ALIGNMENT - 1))
    {
      r = r & (~(MAX_EXTENDED_ALIGNMENT - 1));
      r += MAX_EXTENDED_ALIGNMENT;
    }
  return (char *)r;
}

/* called from kernel setup code.
 * Sets up the actual arguments, except the local ones. */
void
setup_kernel_arg_array (kernel_run_command *k)
{
  struct pocl_argument *al;
  cl_kernel kernel = k->kernel;
  cl_uint i;
  void **arguments;
  void **arguments2;
  k->arguments = arguments
      = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, ARGS_SIZE);
  k->arguments2 = arguments2
      = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, ARGS_SIZE);

  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(k->kernel_args[i]);
      if (kernel->arg_info[i].is_local)
        {
          arguments[i] = NULL;
          arguments2[i] = NULL;
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
      {
        /* It's legal to pass a NULL pointer to clSetKernelArguments. In
           that case we must pass the same NULL forward to the kernel.
           Otherwise, the user must have created a buffer with per device
           pointers stored in the cl_mem. */
        if (al->value == NULL)
          {
            arguments[i] = &arguments2[i];
            arguments2[i] = NULL;
          }
        else
          {
            cl_mem m = *(cl_mem *)al->value;
            if (m->device_ptrs)
              arguments[i] = &(m->device_ptrs[k->device->dev_id].mem_ptr);
            else
              arguments[i] = &(m->mem_host_ptr);
          }
      }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          fill_dev_image_t(&di, al, k->device);
          void *devptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                              sizeof (dev_image_t));
          arguments[i] = &arguments2[i];
          arguments2[i] = devptr;
          memcpy (devptr, &di, sizeof (dev_image_t));
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          fill_dev_sampler_t(&ds, al);

          arguments[i] = &arguments2[i];
          arguments2[i] = (void *)ds;
        }
      else
        arguments[i] = al->value;
    }
}

/* called from each driver thread.
 * "arguments" and "arguments2" are the output:
 * driver-thread-local copies of kern args.
 *
 * they're set up by 1) memcpy from kernel_run_command, 2) all
 * local args are set to thread-local "local memory" storage. */
void
setup_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                    kernel_run_command *k, char *local_mem,
                                    size_t local_mem_size)
{
  cl_kernel kernel = k->kernel;
  cl_uint i;

  memcpy (arguments2, k->arguments2, ARGS_SIZE);
  memcpy (arguments, k->arguments, ARGS_SIZE);

  char *start = local_mem;

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
          size_t size = k->kernel_args[i].size;
          arguments[i] = &arguments2[i];
          arguments2[i] = start;
          start += size;
          start = align_ptr (start);
          assert ((size_t) (start - local_mem) <= local_mem_size);
        }
    }

  /* Allocate the automatic local buffers which are implemented as implicit
     extra arguments at the end of the kernel argument list. */
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      size_t size = k->kernel_args[i].size;
      arguments[i] = &arguments2[i];
      arguments2[i] = start;
      start += size;
      start = align_ptr (start);
      assert ((size_t) (start - local_mem) <= local_mem_size);
    }
}

/* called from kernel teardown code.
 * frees the actual arguments, except the local ones. */
void
free_kernel_arg_array (kernel_run_command *k)
{
  cl_uint i;
  cl_kernel kernel = k->kernel;
  void **arguments = k->arguments;
  void **arguments2 = k->arguments2;

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
          assert (arguments[i] == NULL);
          assert (arguments2[i] == NULL);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          POCL_MEM_FREE (arguments2[i]);
        }
    }

  POCL_MEM_FREE (k->arguments);
  POCL_MEM_FREE (k->arguments2);
}

/* called from each driver thread.
 * frees the local arguments. */
void
free_kernel_arg_array_with_locals (void **arguments, void **arguments2,
                                   kernel_run_command *k)
{
  cl_kernel kernel = k->kernel;
  cl_uint i;

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
          arguments[i] = NULL;
          arguments2[i] = NULL;
        }
    }

  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      arguments[i] = NULL;
      arguments2[i] = NULL;
    }
}
