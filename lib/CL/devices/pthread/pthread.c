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
#include "install-paths.h"
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include "pocl_runtime_config.h"
#include "utlist.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "config.h"
#include "devices.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

#ifdef CUSTOM_BUFFER_ALLOCATOR

#include "bufalloc.h"
#include <../dev_image.h>

/* Instead of mallocing a buffer size for a region, try to allocate 
   this many times the buffer size to hopefully avoid mallocs for 
   the next buffer allocations.
   
   Falls back to single multiple allocation if fails to allocate a
   larger region. */
#define ALLOCATION_MULTIPLE 32

/* To avoid memory hogging in case of larger buffers, limit the
   extra allocation margin to this number of megabytes.

   The extra allocation should be done to avoid repetitive calls and
   memory fragmentation for smaller buffers only. 
 */
#define ADDITIONAL_ALLOCATION_MAX_MB 100

/* Whether to immediately free a region in case the last chunk was
   deallocated. If 0, it can reuse the same region over multiple kernels. */
#define FREE_EMPTY_REGIONS 0

/* CUSTOM_BUFFER_ALLOCATOR */
#endif

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 128

/* The name of the environment variable used to force a certain max thread count
   for the thread execution. */
#define THREAD_COUNT_ENV "POCL_MAX_PTHREAD_COUNT"

typedef struct thread_arguments thread_arguments;
struct thread_arguments 
{
  void *data;
  cl_kernel kernel;
  unsigned device;
  struct pocl_context pc;
  int last_gid_x; 
  pocl_workgroup workgroup;
  struct pocl_argument *kernel_args;
  thread_arguments *volatile next;
};

typedef struct _mem_regions_management{
  ba_lock_t mem_regions_lock;
  struct memory_region *mem_regions;
} mem_regions_management;

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;

#ifdef CUSTOM_BUFFER_ALLOCATOR
  /* Lock for protecting the mem_regions linked list. Held when new mem_regions
     are created or old ones freed. */
  mem_regions_management* mem_regions;
#endif

};


static thread_arguments *volatile thread_argument_pool = 0;
static int argument_pool_initialized = 0;
pocl_lock_t ta_pool_lock;
static int get_max_thread_count();
static void * workgroup_thread (void *p);

void pocl_init_thread_argument_manager (void)
{
  if (!argument_pool_initialized)
    {
      argument_pool_initialized = 1;
      POCL_INIT_LOCK (ta_pool_lock);
    }
}

thread_arguments* new_thread_arguments ()
{
  thread_arguments *ta = NULL;
  POCL_LOCK (ta_pool_lock);
  if (ta = thread_argument_pool)
    {
      LL_DELETE (thread_argument_pool, ta);
      POCL_UNLOCK (ta_pool_lock);
      return ta;
    }
  POCL_UNLOCK (ta_pool_lock);
    
  return calloc (1, sizeof (thread_arguments));
}

void free_thread_arguments (thread_arguments *ta)
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
  ops->init_device_infos = pocl_pthread_init_device_infos;
  ops->uninit = pocl_pthread_uninit;
  ops->init = pocl_pthread_init;
  ops->malloc = pocl_pthread_malloc;
  ops->alloc_mem_obj = pocl_pthread_alloc_mem_obj;
  ops->free = pocl_pthread_free;
  ops->read = pocl_pthread_read;
  ops->write = pocl_pthread_write;
  ops->copy = pocl_pthread_copy;
  ops->copy_rect = pocl_pthread_copy_rect;
  ops->run = pocl_pthread_run;
  ops->compile_submitted_kernels = pocl_basic_compile_submitted_kernels;

}

void
pocl_pthread_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos(dev);

  dev->type = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_DEFAULT;
  /* This could be SIZE_T_MAX, but setting it to INT_MAX should suffice, */
  /* and may avoid errors in user code that uses int instead of size_t */
  dev->max_work_item_sizes[0] = 1024;
  dev->max_work_item_sizes[1] = 1024;
  dev->max_work_item_sizes[2] = 1024;

}

void
pocl_pthread_init (cl_device_id device, const char* parameters)
{
  struct data *d; 
  static mem_regions_management* mrm = NULL;
  static int global_mem_id;
  int i;

  // TODO: this checks if the device was already initialized previously.
  // Should we instead have a separate bool field in device, or do the
  // initialization at library startup time with __attribute__((constructor))?
  if (device->data!=NULL)
    return;  

  d = (struct data *) malloc (sizeof (struct data));
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;
#ifdef CUSTOM_BUFFER_ALLOCATOR  
  if (mrm == NULL)
    {
      mrm = malloc (sizeof (mem_regions_management));
      BA_INIT_LOCK (mrm->mem_regions_lock);
      mrm->mem_regions = NULL;
    }
  d->mem_regions = mrm;
#endif  

  device->address_bits = sizeof(void*) * 8;

  /* Use the minimum values until we get a more sensible 
     upper limit from somewhere. */
  device->max_read_image_args = device->max_write_image_args = 128;
  device->image2d_max_width = device->image2d_max_height = 8192;
  device->image3d_max_width = device->image3d_max_height = device->image3d_max_depth = 2048;
  device->max_samplers = 16;  
  device->max_constant_args = 8;

  device->min_data_type_align_size = device->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;

  /* Note: The specification describes identifiers being delimited by
     only a single space character. Some programs that check the device's
     extension  string assume this rule. Future extension additions should
     ensure that there is no more than a single space between
     identifiers. */

#if SIZEOF_DOUBLE == 8
#define DOUBLE_EXT "cl_khr_fp64 "
#else
#define DOUBLE_EXT 
#endif

#if SIZEOF___FP16 == 2
#define HALF_EXT "cl_khr_fp16 "
#else
#define HALF_EXT
#endif

  device->extensions = DOUBLE_EXT HALF_EXT "cl_khr_byte_addressable_store";

  pocl_cpuinfo_detect_device_info(device);
  pocl_topology_detect_device_info(device);

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
#ifdef CUSTOM_BUFFER_ALLOCATOR
  memory_region_t *region, *temp;
  DL_FOREACH_SAFE(d->mem_regions->mem_regions, region, temp)
    {
      DL_DELETE(d->mem_regions->mem_regions, region);
      free ((void*)region->chunks->start_address);
      free (region);    
    }
  d->mem_regions->mem_regions = NULL;
#endif  
  free (d);
  device->data = NULL;
}


#ifdef CUSTOM_BUFFER_ALLOCATOR
static int
allocate_aligned_buffer (struct data* d, void **memptr, size_t alignment, size_t size) 
{
  BA_LOCK(d->mem_regions->mem_regions_lock);
  chunk_info_t *chunk = alloc_buffer (d->mem_regions->mem_regions, size);
  if (chunk == NULL)
    {
      memory_region_t *new_mem_region = 
        (memory_region_t*)malloc (sizeof (memory_region_t));

      if (new_mem_region == NULL) 
        {
          BA_UNLOCK (d->mem_regions->mem_regions_lock);
          return ENOMEM;
        }

      /* Fallback to the minimum size in case of overflow. 
         Allocate a larger chunk to avoid allocation overheads
         later on. */
      size_t region_size = 
        max(min(size + ADDITIONAL_ALLOCATION_MAX_MB * 1024 * 1024, 
                size * ALLOCATION_MULTIPLE), size);

      assert (region_size >= size);

      void* space = NULL;
      if ((posix_memalign (&space, alignment, region_size)) != 0)
        {
          /* Failed to allocate a large region. Fall back to allocating 
             the smallest possible region for the buffer. */
          if ((posix_memalign (&space, alignment, size)) != 0) 
            {
              BA_UNLOCK (d->mem_regions->mem_regions_lock);
              return ENOMEM;
            }
          region_size = size;
        }

      init_mem_region (new_mem_region, (memory_address_t)space, region_size);
      new_mem_region->alignment = alignment;
      DL_APPEND (d->mem_regions->mem_regions, new_mem_region);
      chunk = alloc_buffer_from_region (new_mem_region, size);

      if (chunk == NULL)
      {
        printf("pocl error: could not allocate a buffer of size %zu from the newly created region of size %zu.\n",
               size, region_size);
        print_chunks(new_mem_region->chunks);
        /* In case the malloc didn't fail it should have been able to allocate 
           the buffer to a newly created Region. */
        assert (chunk != NULL);
      }
    }
  BA_UNLOCK (d->mem_regions->mem_regions_lock);
  
  *memptr = (void*) chunk->start_address;
  return 0;
}

#else

static int
allocate_aligned_buffer (struct data* d, void **memptr, size_t alignment, size_t size) 
{
  return posix_memalign (memptr, alignment, size);
}

#endif

void *
pocl_pthread_malloc (void *device_data, cl_mem_flags flags, size_t size, void *host_ptr)
{
  void *b;
  struct data* d = (struct data*)device_data;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      if (allocate_aligned_buffer (d, &b, MAX_EXTENDED_ALIGNMENT, size) == 0)
        {
          memcpy (b, host_ptr, size);
          return b;
        }
      
      return NULL;
    }
  
  if (flags & CL_MEM_USE_HOST_PTR && host_ptr != NULL)
    {
      return host_ptr;
    }

  if (allocate_aligned_buffer (d, &b, MAX_EXTENDED_ALIGNMENT, size) == 0)
    return b;
  
  return NULL;
}

cl_int
pocl_pthread_alloc_mem_obj (cl_device_id device, cl_mem mem_obj)
{
  void *b = NULL;
  struct data* d = (struct data*)device->data;
  cl_int flags = mem_obj->flags;

  /* if memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
    {
      if (flags & CL_MEM_USE_HOST_PTR && mem_obj->mem_host_ptr != NULL)
        {
          b = mem_obj->mem_host_ptr;
        }
      else if (allocate_aligned_buffer (d, &b, MAX_EXTENDED_ALIGNMENT, 
                                        mem_obj->size) != 0)
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;

      if (flags & CL_MEM_COPY_HOST_PTR)
        memcpy (b, mem_obj->mem_host_ptr, mem_obj->size);
    
      mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
      mem_obj->device_ptrs[device->global_mem_id].global_mem_id = 
        device->global_mem_id;
    }
  /* copy already allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id] = 
    mem_obj->device_ptrs[device->global_mem_id];
    
  return CL_SUCCESS;
}

#ifdef CUSTOM_BUFFER_ALLOCATOR
void
pocl_pthread_free (void *device_data, cl_mem_flags flags, void *ptr)
{
  struct data* d = (struct data*) device_data;
  memory_region_t *region = NULL;

  if (flags & CL_MEM_USE_HOST_PTR)
      return; /* The host code should free the host ptr. */

  region = free_buffer (d->mem_regions->mem_regions, (memory_address_t)ptr);

  assert(region != NULL && "Unable to find the region for chunk.");

#if FREE_EMPTY_REGIONS == 1
  BA_LOCK(d->mem_regions->mem_regions_lock);
  BA_LOCK(region->lock);
  if (region->last_chunk == region->chunks && 
      !region->chunks->is_allocated) 
    {
      /* All chunks have been deallocated. free() the whole 
         memory region at once. */
      DL_DELETE(d->mem_regions->mem_regions, region);
      free ((void*)region->last_chunk->start_address);
      free (region);    
    }  
  BA_UNLOCK(region->lock);
  BA_UNLOCK(d->mem_regions->mem_regions_lock);
#endif
}

#else

void
pocl_pthread_free (void *data, cl_mem_flags flags, void *ptr)
{
  if (flags & CL_MEM_COPY_HOST_PTR)
    return;
  
  free (ptr);
}
#endif

void
pocl_pthread_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
}

void
pocl_pthread_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;
  
  memcpy (device_ptr, host_ptr, cb);
}


void
pocl_pthread_copy (void *data, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb)
{
  if (src_ptr == dst_ptr)
    return;
  
  memcpy (dst_ptr, src_ptr, cb);
}

void
pocl_pthread_copy_rect (void *data,
                        const void *__restrict const src_ptr,
                        void *__restrict__ const dst_ptr,
                        const size_t *__restrict__ const src_origin,
                        const size_t *__restrict__ const dst_origin, 
                        const size_t *__restrict__ const region,
                        size_t const src_row_pitch,
                        size_t const src_slice_pitch,
                        size_t const dst_row_pitch,
                        size_t const dst_slice_pitch)
{
  char const *__restrict const adjusted_src_ptr = 
    (char const*)src_ptr +
    src_origin[0] + src_row_pitch * (src_origin[1] + src_slice_pitch * src_origin[2]);
  char *__restrict__ const adjusted_dst_ptr = 
    (char*)dst_ptr +
    dst_origin[0] + dst_row_pitch * (dst_origin[1] + dst_slice_pitch * dst_origin[2]);
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
              adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
              region[0]);
}

#define FALLBACK_MAX_THREAD_COUNT 8
//#define DEBUG_MT
//#define DEBUG_MAX_THREAD_COUNT
/**
 * Return an estimate for the maximum thread count that should produce
 * the maximum parallelism without extra threading overheads.
 */
static
int 
get_max_thread_count(cl_device_id device) 
{
  /* if return THREAD_COUNT_ENV if set, 
     else return fallback or max_compute_units */
  if (device->max_compute_units == 0)
    return pocl_get_int_option (THREAD_COUNT_ENV, FALLBACK_MAX_THREAD_COUNT);
  else
    return pocl_get_int_option(THREAD_COUNT_ENV, device->max_compute_units);
}

void
pocl_pthread_run 
(void *data, 
 _cl_command_node* cmd)
{
  struct data *d;
  int error;
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  unsigned device;
  cl_device_id device_ptr;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  struct thread_arguments *arguments;
  static int max_threads = 0; /* this needs to be asked only once */

  d = (struct data *) data;

  /* Find which device number within the context correspond
     to current device.  */
  for (i = 0; i < kernel->context->num_devices; ++i)
    {
      if (kernel->context->devices[i]->data == data)
        {
          device = i;
          device_ptr = kernel->context->devices[i];
          break;
        }
    }


  int num_groups_x = pc->num_groups[0];
  /* TODO: distributing the work groups in the x dimension is not always the
     best option. This assumes x dimension has enough work groups to utilize
     all the threads. */
  if (max_threads == 0)
    max_threads = get_max_thread_count(device_ptr);

  int num_threads = min(max_threads, num_groups_x);
  pthread_t *threads = (pthread_t*) malloc (sizeof (pthread_t)*num_threads);
  
  int wgs_per_thread = num_groups_x / num_threads;
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
  
  int first_gid_x = 0;
  int last_gid_x = wgs_per_thread - 1;
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
    arguments->device = device;
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

  free(threads);
}

void *
pocl_pthread_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size, void* host_ptr) 
{
  /* All global pointers of the pthread/CPU device are in 
     the host address space already, and up to date. */     
  return buf_ptr + offset;
}

void *
workgroup_thread (void *p)
{
  struct thread_arguments *ta = (struct thread_arguments *) p;
  void *arguments[ta->kernel->num_args + ta->kernel->num_locals];
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
      if (kernel->arg_is_local[i])
        {
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_pthread_malloc(ta->data, 0, al->size, NULL);
        }
      else if (kernel->arg_is_pointer[i])
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
            arguments[i] = 
              &((*(cl_mem *)(al->value))->device_ptrs[ta->device].mem_ptr);
          }
      }
      else if (kernel->arg_is_image[i])
        {
          dev_image_t di;
          fill_dev_image_t(&di, al, ta->device);
          void* devptr = pocl_pthread_malloc(ta->data, 0, sizeof(dev_image_t), NULL);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;       
          pocl_pthread_write (ta->data, &di, devptr, sizeof(dev_image_t));
        }
      else if (kernel->arg_is_sampler[i])
        {
          dev_sampler_t ds;
          
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_pthread_malloc 
            (ta->data, 0, sizeof(dev_sampler_t), NULL);
          pocl_pthread_write (ta->data, &ds, *(void**)arguments[i], 
                              sizeof(dev_sampler_t));
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
      *(void **)(arguments[i]) = pocl_pthread_malloc (ta->data, 0, al->size, 
                                                      NULL);
    }

  int first_gid_x = ta->pc.group_id[0];
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
      if (kernel->arg_is_local[i] )
        {
          pocl_pthread_free (ta->data, 0, *(void **)(arguments[i]));
          free (arguments[i]);
        }
      else if (kernel->arg_is_sampler[i] || kernel->arg_is_image[i] || 
               (kernel->arg_is_pointer[i] && *(void**)arguments[i] == NULL))
        {
          free (arguments[i]);
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      pocl_pthread_free (ta->data, 0, *(void **)(arguments[i]));
      free (arguments[i]);
    }
  free_thread_arguments (ta);

  return NULL;
}
