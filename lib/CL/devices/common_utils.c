/* common_utils.c - common utilities for pthread and tbb devices

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2019 Pekka Jääskeläinen and
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

#include <string.h>

#include "common.h"
#include "common_utils.h"
#include "pocl_mem_management.h"
#include "utlist.h"

/* NOTE: k->lock is probably unnecessary for the tbb device */
#ifdef USE_POCL_MEMMANAGER

static kernel_run_command *volatile kernel_pool = 0;
static int kernel_pool_initialized = 0;
static pocl_lock_t kernel_pool_lock;


void pocl_init_kernel_run_command_manager ()
{
  if (!kernel_pool_initialized)
    {
      kernel_pool_initialized = 1;
      POCL_INIT_LOCK (kernel_pool_lock);
    }
}

void pocl_init_thread_argument_manager ()
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
      PTHREAD_CHECK (pthread_mutex_init (&k->lock, NULL));
      POCL_UNLOCK (kernel_pool_lock);
      return k;
    }

  POCL_UNLOCK (kernel_pool_lock);
  k = (kernel_run_command*)calloc (1, sizeof (kernel_run_command));
  PTHREAD_CHECK (pthread_mutex_init (&k->lock, NULL));
  return k;
}

void free_kernel_run_command (kernel_run_command *k)
{
  POCL_LOCK (kernel_pool_lock);
  PTHREAD_CHECK (pthread_mutex_destroy (&k->lock));
  LL_PREPEND (kernel_pool, k);
  POCL_UNLOCK (kernel_pool_lock);
}

#endif

#define ARGS_SIZE (sizeof (void *) * (meta->num_args + meta->num_locals + 1))

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

  pocl_kernel_metadata_t *meta = k->kernel->meta;
  cl_uint i;
  void **arguments;
  void **arguments2;
  k->arguments = arguments
      = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, ARGS_SIZE);
  k->arguments2 = arguments2
      = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, ARGS_SIZE);

  for (i = 0; i < meta->num_args; ++i)
    {
      al = &(k->kernel_args[i]);
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          arguments[i] = NULL;
          arguments2[i] = NULL;
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          arguments[i] = &arguments2[i];
          if (al->value == NULL)
            {
              arguments2[i] = NULL;
            }
          else
            {
              void *ptr = NULL;
              if (al->is_svm)
                {
                  ptr = *(void **)al->value;
                }
              else
                {
                  cl_mem m = (*(cl_mem *)(al->value));
                  ptr = m->device_ptrs[k->device->global_mem_id].mem_ptr;
                }
              arguments2[i] = (char *)ptr + al->offset;
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          pocl_fill_dev_image_t (&di, al, k->device);
          void *devptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                              sizeof (dev_image_t));
          arguments[i] = &arguments2[i];
          arguments2[i] = devptr;
          memcpy (devptr, &di, sizeof (dev_image_t));
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          pocl_fill_dev_sampler_t (&ds, al);

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
  pocl_kernel_metadata_t *meta = k->kernel->meta;
  cl_uint i;

  memcpy (arguments2, k->arguments2, ARGS_SIZE);
  memcpy (arguments, k->arguments, ARGS_SIZE);

  char *start = local_mem;

  for (i = 0; i < meta->num_args; ++i)
    {
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          size_t size = k->kernel_args[i].size;
          if (!k->device->device_alloca_locals)
            {
              arguments[i] = &arguments2[i];
              arguments2[i] = start;
              start += size;
              start = align_ptr (start);
              assert ((size_t) (start - local_mem) <= local_mem_size);
            }
          else
            {
              /* Local buffers are allocated in the device side work-group
                 launcher. Let's pass only the sizes of the local args in
                 the arg buffer. */
              assert (sizeof (size_t) == sizeof (void *));
              arguments[i] = (void *)size;
            }
        }
    }
  if (!k->device->device_alloca_locals)
    /* Allocate the automatic local buffers which are implemented as implicit
       extra arguments at the end of the kernel argument list. */
    for (i = 0; i < meta->num_locals; ++i)
      {
        cl_uint j = meta->num_args + i;
        size_t size = meta->local_sizes[i];
        arguments[j] = &arguments2[j];
        arguments2[j] = start;
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
  pocl_kernel_metadata_t *meta = k->kernel->meta;
  void **arguments = k->arguments;
  void **arguments2 = k->arguments2;

  for (i = 0; i < meta->num_args; ++i)
    {
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          if (!k->device->device_alloca_locals)
            {
              assert (arguments[i] == NULL);
              assert (arguments2[i] == NULL);
            }
          else
            {
              /* Device side local space allocation has deallocation via stack
                 unwind. */
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
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
  pocl_kernel_metadata_t *meta = k->kernel->meta;
  cl_uint i;

  for (i = 0; i < meta->num_args; ++i)
    {
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          arguments[i] = NULL;
          arguments2[i] = NULL;
        }
    }

  for (i = 0; i < meta->num_locals; ++i)
    {
      arguments[meta->num_args + i] = NULL;
      arguments2[meta->num_args + i] = NULL;
    }
}
