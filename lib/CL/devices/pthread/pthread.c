/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos, 
   Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "pthread.h"
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include "utlist.h"

#include "config.h"

#ifdef CUSTOM_BUFFER_ALLOCATOR

#include "bufalloc.h"

/* Instead of mallocing a buffer size for a region, try to allocate 
   this many times the buffer size to hopefully avoid mallocs for 
   the next buffer allocations.
   
   Falls back to single multiple allocation if fails to allocate a
   larger region. */
#define ALLOCATION_MULTIPLE 32

/* Whether to immediately free a region in case the last chunk was
   deallocated. If 0, it can reuse the same region over multiple kernels. */
#define FREE_EMPTY_REGIONS 0

/* CUSTOM_BUFFER_ALLOCATOR */
#endif

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128

#define ALIGNMENT (max(ALIGNOF_FLOAT16, ALIGNOF_DOUBLE16))

/* The name of the environment variable used to force a certain max thread count
   for the thread execution. */
#define THREAD_COUNT_ENV "POCL_MAX_PTHREAD_COUNT"

struct thread_arguments {
  void *data;
  cl_kernel kernel;
  unsigned device;
  struct pocl_context pc;
  int last_gid_x; 
  pocl_workgroup workgroup;
};

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;

#ifdef CUSTOM_BUFFER_ALLOCATOR
  /* Lock for protecting the mem_regions linked list. Held when new mem_regions
     are created or old ones freed. */
  ba_lock_t mem_regions_lock;
  struct memory_region *mem_regions;
#endif

};

static void * workgroup_thread (void *p);

/* This could be SIZE_T_MAX, but setting it to INT_MAX should suffice,
   and may avoid errors in user code that uses int instead of
   size_t */
size_t pocl_pthread_max_work_item_sizes[] = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX};

void
pocl_pthread_init (cl_device_id device)
{
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
  device->max_compute_units = get_max_thread_count();
#ifdef CUSTOM_BUFFER_ALLOCATOR  
  BA_INIT_LOCK (d->mem_regions_lock);
  d->mem_regions = NULL;
#endif  
}

void
pocl_pthread_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
#ifdef CUSTOM_BUFFER_ALLOCATOR
  memory_region_t *region, *temp;
  DL_FOREACH_SAFE(d->mem_regions, region, temp)
    {
      DL_DELETE(d->mem_regions, region);
      free ((void*)region->chunks->start_address);
      free (region);    
    }
  d->mem_regions = NULL;
#endif  
  free (d);
  device->data = NULL;
}


#ifdef CUSTOM_BUFFER_ALLOCATOR
static int
allocate_aligned_buffer (struct data* d, void **memptr, size_t alignment, size_t size) 
{
  BA_LOCK(d->mem_regions_lock);
  chunk_info_t *chunk = alloc_buffer (d->mem_regions, size);
  if (chunk == NULL)
    {
      memory_region_t *new_mem_region = 
        (memory_region_t*)malloc (sizeof (memory_region_t));

      if (new_mem_region == NULL) 
        {
          BA_UNLOCK (d->mem_regions_lock);
          return ENOMEM;
        }

      size_t region_size = size*ALLOCATION_MULTIPLE;
      void* space = NULL;
      if ((posix_memalign (&space, alignment, region_size)) != 0)
        {
          /* Failed to allocate a large region. Fall back to allocating 
             the smallest possible region for the buffer. */
          if ((posix_memalign (&space, alignment, size)) != 0) 
            {
              BA_UNLOCK (d->mem_regions_lock);
              return ENOMEM;
            }
          region_size = size;
        }

      init_mem_region (new_mem_region, (memory_address_t)space, region_size);
      new_mem_region->alignment = alignment;
      DL_APPEND (d->mem_regions, new_mem_region);
      chunk = alloc_buffer_from_region (new_mem_region, size);
      
      /* In case the malloc didn't fail it should have been able to allocate 
         the buffer to a newly created Region. */
      assert (chunk != NULL);
    }
  BA_UNLOCK (d->mem_regions_lock);
  
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
      if (allocate_aligned_buffer (d, &b, ALIGNMENT, size) == 0)
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

  if (allocate_aligned_buffer (d, &b, ALIGNMENT, size) == 0)
    return b;
  
  return NULL;
}

#ifdef CUSTOM_BUFFER_ALLOCATOR
void
pocl_pthread_free (void *device_data, cl_mem_flags flags, void *ptr)
{
  struct data* d = (struct data*) device_data;
  memory_region_t *region = NULL;

  if (flags & CL_MEM_USE_HOST_PTR)
      return; /* The host code should free the host ptr. */

  region = free_buffer (d->mem_regions, (memory_address_t)ptr);

  assert(region != NULL && "Unable to find the region for chunk.");

#if FREE_EMPTY_REGIONS == 1
  BA_LOCK(d->mem_regions_lock);
  BA_LOCK(region->lock);
  if (region->last_chunk == region->chunks && 
      !region->chunks->is_allocated) 
    {
      /* All chunks have been deallocated. free() the whole 
         memory region at once. */
      DL_DELETE(d->mem_regions, region);
      free ((void*)region->last_chunk->start_address);
      free (region);    
    }  
  BA_UNLOCK(region->lock);
  BA_UNLOCK(d->mem_regions_lock);
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
pocl_pthread_read_rect (void *data,
                        void *__restrict__ const host_ptr,
                        void *__restrict__ const device_ptr,
                        const size_t *__restrict__ const buffer_origin,
                        const size_t *__restrict__ const host_origin, 
                        const size_t *__restrict__ const region,
                        size_t const buffer_row_pitch,
                        size_t const buffer_slice_pitch,
                        size_t const host_row_pitch,
                        size_t const host_slice_pitch)
{
  char const *__restrict const adjusted_device_ptr = 
    (char const*)device_ptr +
    buffer_origin[0] + buffer_row_pitch * (buffer_origin[1] + buffer_slice_pitch * buffer_origin[2]);
  char *__restrict__ const adjusted_host_ptr = 
    (char*)host_ptr +
    host_origin[0] + host_row_pitch * (host_origin[1] + host_slice_pitch * host_origin[2]);
  
  size_t j, k;
  
  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              region[0]);
}

void
pocl_pthread_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;
  
  memcpy (device_ptr, host_ptr, cb);
}

void
pocl_pthread_write_rect (void *data,
                         const void *__restrict__ const host_ptr,
                         void *__restrict__ const device_ptr,
                         const size_t *__restrict__ const buffer_origin,
                         const size_t *__restrict__ const host_origin, 
                         const size_t *__restrict__ const region,
                         size_t const buffer_row_pitch,
                         size_t const buffer_slice_pitch,
                         size_t const host_row_pitch,
                         size_t const host_slice_pitch)
{
  char *__restrict const adjusted_device_ptr = 
    (char*)device_ptr +
    buffer_origin[0] + buffer_row_pitch * (buffer_origin[1] + buffer_slice_pitch * buffer_origin[2]);
  char const *__restrict__ const adjusted_host_ptr = 
    (char const*)host_ptr +
    host_origin[0] + host_row_pitch * (host_origin[1] + host_slice_pitch * host_origin[2]);
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              region[0]);
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

//#define DEBUG_MT
//#define DEBUG_MAX_THREAD_COUNT
/**
 * Return an estimate for the maximum thread count that should produce
 * the maximum parallelism without extra threading overheads.
 */
int 
get_max_thread_count() 
{
  /* query from /proc/cpuinfo how many hardware threads there are, if available */
  const char* cpuinfo = "/proc/cpuinfo";
  /* eight is a good round number ;) */
  const int FALLBACK_MAX_THREAD_COUNT = 8;

  static int cores = 0;
  if (cores != 0)
      return cores;

  if (getenv(THREAD_COUNT_ENV) != NULL) 
    {
      cores = atoi(getenv(THREAD_COUNT_ENV));
      return cores;
    }

  if (access (cpuinfo, R_OK) == 0) 
    {
      FILE *f = fopen (cpuinfo, "r");
#     define MAX_CPUINFO_SIZE 64*1024
      char contents[MAX_CPUINFO_SIZE];
      int num_read = fread (contents, 1, MAX_CPUINFO_SIZE - 1, f);            
      fclose (f);
      contents[num_read] = '\0';

      /* Count the number of times 'processor' keyword is found which
         should give the number of cores overall in a multiprocessor
         system. In Meego Harmattan on ARM it prints Processor instead of
         processor */
      cores = 0;
      char* p = contents;
      while ((p = strstr (p, "rocessor")) != NULL) 
        {
          cores++;
          /* Skip to the end of the line. Otherwise causes two cores
             to be detected in case of, for example:
             Processor       : ARMv7 Processor rev 2 (v7l) */
          char* eol = strstr (p, "\n");
          if (eol != NULL)
              p = eol;
          ++p;
        }     
#ifdef DEBUG_MAX_THREAD_COUNT 
      printf("total cores %d\n", cores);
#endif
      if (cores == 0)
        return FALLBACK_MAX_THREAD_COUNT;

      int cores_per_cpu = 1;
      p = contents;
      if ((p = strstr (p, "cpu cores")) != NULL)
        {
          if (sscanf (p, ": %d\n", &cores_per_cpu) != 1)
            cores_per_cpu = 1;
#ifdef DEBUG_MAX_THREAD_COUNT 
          printf ("cores per cpu %d\n", cores_per_cpu);
#endif
        }

      int siblings = 1;
      p = contents;
      if ((p = strstr (p, "siblings")) != NULL)
        {
          if (sscanf (p, ": %d\n", &siblings) != 1)
            siblings = cores_per_cpu;
#ifdef DEBUG_MAX_THREAD_COUNT 
          printf ("siblings %d\n", siblings);
#endif
        }
      if (siblings > cores_per_cpu) {
#ifdef DEBUG_MAX_THREAD_COUNT 
        printf ("max threads %d\n", cores*(siblings/cores_per_cpu));
#endif
        return cores*(siblings/cores_per_cpu); /* hardware threading is on */
      } else {
#ifdef DEBUG_MAX_THREAD_COUNT 
        printf ("max threads %d\n", cores);
#endif
        return cores; /* only multicore, if not unicore*/
      }      
    } 
#ifdef DEBUG_MAX_THREAD_COUNT 
  printf ("could not open /proc/cpuinfo, falling back to max threads %d\n", 
          FALLBACK_MAX_THREAD_COUNT);
#endif
  return FALLBACK_MAX_THREAD_COUNT;
}

void
pocl_pthread_run (void *data, const char *parallel_filename,
		 cl_kernel kernel,
		 struct pocl_context *pc)
{
  struct data *d;
  char template[] = ".naruXXXXXX";
  char *tmpdir;
  int error;
  char bytecode[POCL_FILENAME_LENGTH];
  char assembly[POCL_FILENAME_LENGTH];
  char module[POCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  unsigned device;
  size_t x, y, z;
  unsigned i;
  pocl_workgroup w;

  d = (struct data *) data;

  if (d->current_kernel != kernel)
    {
      tmpdir = mkdtemp (template);
      assert (tmpdir != NULL);
      
      error = snprintf (bytecode, POCL_FILENAME_LENGTH,
			"%s/parallel.bc",
			tmpdir);
      assert (error >= 0);
      
      error = snprintf (command, COMMAND_LENGTH,
			LLVM_LD " -link-as-library -o %s %s",
			bytecode,
			parallel_filename);
      assert (error >= 0);
      
      error = system(command);
      assert (error == 0);
      
      error = snprintf (assembly, POCL_FILENAME_LENGTH,
			"%s/parallel.s",
			tmpdir);
      assert (error >= 0);
      
      // "-relocation-model=dynamic-no-pic" is a magic string,
      // I do not know why it has to be there to produce valid
      // sos on x86_64
      error = snprintf (command, COMMAND_LENGTH,
			LLC " " HOST_LLC_FLAGS " -o %s %s",
			assembly,
			bytecode);
      assert (error >= 0);
      
      error = system (command);
      assert (error == 0);
      
      error = snprintf (module, POCL_FILENAME_LENGTH,
			"%s/parallel.so",
			tmpdir);
      assert (error >= 0);
      
      error = snprintf (command, COMMAND_LENGTH,
			CLANG " -c -o %s.o %s",
			module,
			assembly);
      assert (error >= 0);
      
      error = system (command);
      assert (error == 0);

      error = snprintf (command, COMMAND_LENGTH,
                       "ld " HOST_LD_FLAGS " -o %s %s.o",
                       module,
                       module);
      assert (error >= 0);

      error = system (command);
      assert (error == 0);
      
      d->current_dlhandle = lt_dlopen (module);
      if (d->current_dlhandle == NULL)
        {
          printf ("pocl error: lt_dlopen(\"%s\") failed with '%s'.\n", module, lt_dlerror());
          printf ("note: missing symbols in the kernel binary might be reported as 'file not found' errors.\n");
          abort();
        }

      d->current_kernel = kernel;
    }

  /* Find which device number within the context correspond
     to current device.  */
  for (i = 0; i < kernel->context->num_devices; ++i)
    {
      if (kernel->context->devices[i]->data == data)
        {
          device = i;
          break;
        }
    }

  snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
	    "_%s_workgroup", kernel->function_name);
  
  w = (pocl_workgroup) lt_dlsym (d->current_dlhandle, workgroup_string);
  assert (w != NULL);
  int num_groups_x = pc->num_groups[0];
  /* TODO: distributing the work groups in the x dimension is not always the
     best option. This assumes x dimension has enough work groups to utilize
     all the threads. */
  int max_threads = get_max_thread_count();
  int num_threads = min(max_threads, num_groups_x);
  pthread_t *threads = (pthread_t*) malloc (sizeof (pthread_t)*num_threads);
  struct thread_arguments *arguments = 
    (struct thread_arguments*) malloc (sizeof (struct thread_arguments)*num_threads);

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

    arguments[i].data = data;
    arguments[i].kernel = kernel;
    arguments[i].device = device;
    arguments[i].pc = *pc;
    arguments[i].pc.group_id[0] = first_gid_x;
    arguments[i].workgroup = w;
    arguments[i].last_gid_x = last_gid_x;

    error = pthread_create (&threads[i],
                            NULL,
                            workgroup_thread,
                            &arguments[i]);
    assert(!error);
  }

  for (i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
#ifdef DEBUG_MT       
    printf("### thread %u finished\n", (unsigned)threads[i]);
#endif
  }

  free(threads);
  free(arguments);
}

void *
pocl_pthread_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size) 
{
  /* All global pointers of the pthread/CPU device are in 
     the host address space already, and up to date. */     
  return buf_ptr + offset;
}

void *
workgroup_thread (void *p)
{
  struct thread_arguments *ta = (struct thread_arguments *) p;
  void *arguments[ta->kernel->num_args];
  struct pocl_argument *al;  
  unsigned i = 0;

  cl_kernel kernel = ta->kernel;
  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(kernel->arguments[i]);
      if (kernel->arg_is_local[i])
        {
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_pthread_malloc(ta->data, 0, al->size, NULL);
        }
      else if (kernel->arg_is_pointer[i])
        arguments[i] = &((*(cl_mem *) (al->value))->device_ptrs[ta->device]);
      else
        arguments[i] = al->value;
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      al = &(kernel->arguments[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_pthread_malloc(ta->data, 0, al->size, NULL);
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
      if (kernel->arg_is_local[i])
        {
          pocl_pthread_free(ta->data, 0, *(void **)(arguments[i]));
          free(arguments[i]);
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      pocl_pthread_free(ta->data, 0, *(void **)(arguments[i]));
      free(arguments[i]);
    }
  
  return NULL;
}
