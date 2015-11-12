/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and
                           Pekka Jääskeläinen / Tampere Univ. of Technology
   
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

#include "pocl-pthread.h"
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "pocl_runtime_config.h"
#include "utlist.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "config.h"
#include "devices.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 1024

/* The name of the environment variable used to force a certain max thread count
   for the thread execution. */
#define THREAD_COUNT_ENV "POCL_MAX_PTHREAD_COUNT"

typedef struct thread_arguments thread_arguments;
struct thread_arguments 
{
  void *data;
  cl_kernel kernel;
  cl_device_id device;
  struct pocl_context pc;
  unsigned last_gid_x;
  pocl_workgroup workgroup;
  struct pocl_argument *kernel_args;
  thread_arguments *volatile next;
};


struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
};


static thread_arguments *volatile thread_argument_pool = 0;
static int argument_pool_initialized = 0;
pocl_lock_t ta_pool_lock;
static size_t get_max_thread_count(cl_device_id device);
static void * workgroup_thread (void *p);

static void pocl_init_thread_argument_manager (void)
{
  if (!argument_pool_initialized)
    {
      argument_pool_initialized = 1;
      POCL_INIT_LOCK (ta_pool_lock);
    }
}

static thread_arguments* new_thread_arguments ()
{
  thread_arguments *ta = NULL;
  POCL_LOCK (ta_pool_lock);
  if ((ta = thread_argument_pool))
    {
      LL_DELETE (thread_argument_pool, ta);
      POCL_UNLOCK (ta_pool_lock);
      return ta;
    }
  POCL_UNLOCK (ta_pool_lock);

  return (thread_arguments*)calloc (1, sizeof (thread_arguments));
}

static void free_thread_arguments (thread_arguments *ta)
{
  POCL_LOCK (ta_pool_lock);
  LL_PREPEND (thread_argument_pool, ta);
  POCL_UNLOCK (ta_pool_lock);
}

void
pocl_pthread_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "pthread";

  /* implementation */
  ops->probe = pocl_pthread_probe;
  ops->init_device_infos = pocl_pthread_init_device_infos;
  ops->uninit = pocl_pthread_uninit;
  ops->init = pocl_pthread_init;
  ops->alloc_mem_obj = pocl_pthread_alloc_mem_obj;
  ops->free = pocl_pthread_free;
  ops->read = pocl_pthread_read;
  ops->write = pocl_pthread_write;
  ops->copy = pocl_pthread_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->run = pocl_pthread_run;
  ops->compile_submitted_kernels = pocl_basic_compile_submitted_kernels;

}

unsigned int
pocl_pthread_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);
  /* Env was not specified, default behavior was to use 1 pthread device */
  if(env_count < 0)
    return 1;

  return env_count;
}

void
pocl_pthread_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos(dev);

  dev->type = CL_DEVICE_TYPE_CPU;
  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1] =
	  dev->max_work_item_sizes[2] = dev->max_work_group_size;

}

void
pocl_pthread_init (cl_device_id device, const char* parameters)
{
  static int device_number = 0;
  struct data *d; 

  // TODO: this checks if the device was already initialized previously.
  // Should we instead have a separate bool field in device, or do the
  // initialization at library startup time with __attribute__((constructor))?
  if (device->data!=NULL)
    return;  

  d = (struct data *) malloc (sizeof (struct data));
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;

  device->address_bits = sizeof(void*) * 8;

  device->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT; // this is in bytes
  device->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT*8; // this is in bits

  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout 
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put 
     a nonzero there for now. */
  device->global_mem_size = 1;
  pocl_topology_detect_device_info(device);
  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  /* in case hwloc doesn't provide a PCI ID, let's generate
     a vendor id that hopefully is unique across vendors. */
  const char *magic = "pocl";
  if (device->vendor_id == 0)
    device->vendor_id =
      magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24;

  device->vendor_id += device_number;
  device_number++;

  // pthread has elementary partitioning support
  device->max_sub_devices = device->max_compute_units;
  device->num_partition_properties = 2;
  device->partition_properties = calloc(device->num_partition_properties,
    sizeof(cl_device_partition_property));
  device->partition_properties[0] = CL_DEVICE_PARTITION_EQUALLY;
  device->partition_properties[1] = CL_DEVICE_PARTITION_BY_COUNTS;
  device->num_partition_types = 0;
  device->partition_type = NULL;

  if(!strcmp(device->llvm_cpu, "(unknown)"))
    device->llvm_cpu = NULL;

  // work-around LLVM bug where sizeof(long)=4
  #ifdef _CL_DISABLE_LONG
  device->has_64bit_long=0;
  #endif

  pocl_init_thread_argument_manager();
}

void
pocl_pthread_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  POCL_MEM_FREE(d);
  device->data = NULL;
}


cl_int
pocl_pthread_alloc_mem_obj (cl_device_id device, cl_mem mem_obj)
{
  void *b = NULL;
  cl_mem_flags flags = mem_obj->flags;

  /* if memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
    {
      if (flags & CL_MEM_USE_HOST_PTR)
        {
          // mem_host_ptr must be non-NULL
          assert(mem_obj->mem_host_ptr != NULL);
          b = mem_obj->mem_host_ptr;
        }
      else
        {
          b = pocl_memalign_alloc_global_mem( device, MAX_EXTENDED_ALIGNMENT,
                                        mem_obj->size);
          if (b==NULL)
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      if (flags & CL_MEM_COPY_HOST_PTR)
        {
          // mem_host_ptr must be non-NULL
          assert(mem_obj->mem_host_ptr != NULL);
          memcpy (b, mem_obj->mem_host_ptr, mem_obj->size);
        }

      mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
      mem_obj->device_ptrs[device->global_mem_id].global_mem_id = 
        device->global_mem_id;
    }
  /* copy already allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id] = 
    mem_obj->device_ptrs[device->global_mem_id];

  return CL_SUCCESS;
}


void
pocl_pthread_free (cl_device_id device, cl_mem memobj)
{
  cl_mem_flags flags = memobj->flags;

  if (flags & CL_MEM_USE_HOST_PTR)
    return;

  void* ptr = memobj->device_ptrs[device->dev_id].mem_ptr;
  size_t size = memobj->size;

  pocl_free_global_mem(device, ptr, size);
}


void
pocl_pthread_read (void *data, void *host_ptr, const void *device_ptr, 
                   size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char*)device_ptr + offset, cb);
}

void
pocl_pthread_write (void *data, const void *host_ptr, void *device_ptr, 
                    size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;
  
  memcpy ((char*)device_ptr + offset, host_ptr, cb);
}

void
pocl_pthread_copy (void *data, const void *src_ptr, size_t src_offset, 
                   void *__restrict__ dst_ptr, size_t dst_offset, size_t cb)
{
  if (src_ptr == dst_ptr)
    return;
  
  memcpy ((char*)dst_ptr + dst_offset, (char*)src_ptr + src_offset, cb);
}

#define FALLBACK_MAX_THREAD_COUNT 8
//#define DEBUG_MT
//#define DEBUG_MAX_THREAD_COUNT
/**
 * Return an estimate for the maximum thread count that should produce
 * the maximum parallelism without extra threading overheads.
 */
static
size_t
get_max_thread_count(cl_device_id device) 
{
  /* if return THREAD_COUNT_ENV if set, 
     else return fallback or max_compute_units */
  if (device->max_compute_units == 0)
    return pocl_get_int_option (THREAD_COUNT_ENV, FALLBACK_MAX_THREAD_COUNT);
  else
    return pocl_get_int_option(THREAD_COUNT_ENV, POCL_REAL_DEV(device)->max_compute_units);
}

void
pocl_pthread_run 
(void *data, 
 _cl_command_node* cmd)
{
  int error;
  unsigned i, max_threads;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  struct thread_arguments *arguments;
  static unsigned default_max_threads = 0; /* this needs to be asked only once */

  size_t num_groups_x = pc->num_groups[0];
  /* TODO: distributing the work groups in the x dimension is not always the
     best option. This assumes x dimension has enough work groups to utilize
     all the threads. */
  if (default_max_threads == 0)
    default_max_threads = get_max_thread_count(cmd->device);

  if (cmd->device->parent_device)
    max_threads = cmd->device->max_compute_units;
  else
    max_threads = default_max_threads;

  unsigned num_threads = min(max_threads, num_groups_x);
  pthread_t *threads = (pthread_t*) malloc (sizeof (pthread_t)*num_threads);
  
  unsigned wgs_per_thread = num_groups_x / num_threads;
  /* In case the work group count is not divisible by the
     number of threads, we have to execute the remaining
     workgroups in one of the threads. */
  /* TODO: This is inefficient; it is better to round up when
     calculating wgs_per_thread */
  int leftover_wgs = num_groups_x - (num_threads*wgs_per_thread);

#ifdef DEBUG_MT    
  printf("### creating %d work group threads\n", num_threads);
  printf("### wgs per thread==%d leftover wgs==%d\n", wgs_per_thread, leftover_wgs);
#endif
  
  unsigned first_gid_x = 0;
  unsigned last_gid_x = wgs_per_thread - 1;
  for (i = 0; i < num_threads; 
       ++i, first_gid_x += wgs_per_thread, last_gid_x += wgs_per_thread) {

    if (i + 1 == num_threads) last_gid_x += leftover_wgs;

#ifdef DEBUG_MT       
    printf("### creating wg thread: first_gid_x==%d, last_gid_x==%d\n",
           first_gid_x, last_gid_x);
#endif
    arguments = new_thread_arguments();
    arguments->data = data;
    arguments->kernel = kernel;
    arguments->device = cmd->device;
    arguments->pc = *pc;
    arguments->pc.group_id[0] = first_gid_x;
    arguments->workgroup = cmd->command.run.wg;
    arguments->last_gid_x = last_gid_x;
    arguments->kernel_args = cmd->command.run.arguments;

    /* TODO: pool of worker threads to avoid syscalls here */
    error = pthread_create (&threads[i],
                            NULL,
                            workgroup_thread,
                            arguments);
    assert(!error);
  }

  for (i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
#ifdef DEBUG_MT       
    printf("### thread %u finished\n", (unsigned)threads[i]);
#endif
  }

  POCL_MEM_FREE(threads);
}

void *
pocl_pthread_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size, void* host_ptr) 
{
  /* All global pointers of the pthread/CPU device are in 
     the host address space already, and up to date. */     
  return (char*)buf_ptr + offset;
}

void *
workgroup_thread (void *p)
{
  struct thread_arguments *ta = (struct thread_arguments *) p;
  void **arguments = (void**)alloca((ta->kernel->num_args + ta->kernel->num_locals)*sizeof(void*));
  struct pocl_argument *al;  
  unsigned i = 0;

  /* TODO: refactor this to share code with basic.c 

     To function 
     void setup_kernel_arg_array(void **arguments, cl_kernel kernel)
     or similar
  */
  cl_kernel kernel = ta->kernel;
  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(ta->kernel_args[i]);
      if (kernel->arg_info[i].is_local)
        {
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, al->size);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
      {
        /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
           that case we must pass the same NULL forward to the kernel.
           Otherwise, the user must have created a buffer with per device
           pointers stored in the cl_mem. */
        if (al->value == NULL) 
          {
            arguments[i] = malloc (sizeof (void *));
            *(void **)arguments[i] = NULL;
          }
        else
          {
            cl_mem m = *(cl_mem *)al->value;
            if (m->device_ptrs)
              arguments[i] = &(m->device_ptrs[ta->device->dev_id].mem_ptr);
            else
              arguments[i] = &(m->mem_host_ptr);
          }
      }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          fill_dev_image_t(&di, al, ta->device);
          void* devptr = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, sizeof(dev_image_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;       
          pocl_pthread_write (ta->data, &di, devptr, 0, sizeof(dev_image_t));
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          fill_dev_sampler_t(&ds, al);
          
          void* devptr = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, sizeof(dev_sampler_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          pocl_pthread_write (ta->data, &ds, devptr, 0, sizeof(dev_sampler_t));
        }
      else
        arguments[i] = al->value;
    }

  /* Allocate the automatic local buffers which are implemented as implicit
     extra arguments at the end of the kernel argument list. */
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      al = &(ta->kernel_args[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_memalign_alloc (MAX_EXTENDED_ALIGNMENT, al->size);
    }

  size_t first_gid_x = ta->pc.group_id[0];
  unsigned gid_z, gid_y, gid_x;
  for (gid_z = 0; gid_z < ta->pc.num_groups[2]; ++gid_z)
    {
      for (gid_y = 0; gid_y < ta->pc.num_groups[1]; ++gid_y)
        {
          for (gid_x = first_gid_x; gid_x <= ta->last_gid_x; ++gid_x)
            {
              ta->pc.group_id[0] = gid_x;
              ta->pc.group_id[1] = gid_y;
              ta->pc.group_id[2] = gid_z;
              ta->workgroup (arguments, &(ta->pc));              
            }
        }
    }

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local )
        {
          POCL_MEM_FREE(*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
                kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_MEM_FREE(*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER && *(void**)arguments[i] == NULL)
        {
          POCL_MEM_FREE(arguments[i]);
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      POCL_MEM_FREE(*(void **)(arguments[i]));
      POCL_MEM_FREE(arguments[i]);
    }
  free_thread_arguments (ta);

  return NULL;
}
