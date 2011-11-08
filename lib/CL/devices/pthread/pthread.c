/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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
#include <unistd.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128

struct pointer_list {
  void *pointer;
  struct pointer_list *next;
};

struct thread_arguments {
  void *data;
  cl_kernel kernel;
  unsigned device;
  struct pocl_context pc;
  int last_gid_x; 
  workgroup workgroup;
};

struct data {
  /* Buffers where host pointer is used, and thus
     should not be deallocated on free. */
  struct pointer_list *host_buffers;
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
};

static void * workgroup_thread (void *p);

size_t pocl_pthread_max_work_item_sizes[] = {1};

void
pocl_pthread_init (cl_device_id device)
{
  struct data *d;
  
  d = (struct data *) malloc (sizeof (struct data));
  
  d->host_buffers = NULL;
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;
}

void *
pocl_pthread_malloc (void *data, cl_mem_flags flags,
		    size_t size, void *host_ptr)
{
  struct data *d;
  void *b;
  struct pointer_list *p;

  d = (struct data *) data;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      b = malloc (size);
      memcpy (b, host_ptr, size);
      
      return b;
    }

  if (host_ptr != NULL)
    {
      if (d->host_buffers == NULL)
        d->host_buffers = malloc (sizeof (struct pointer_list));
      
      p = d->host_buffers;
      while (p->next != NULL)
        p = p->next;

      p->next = malloc (sizeof (struct pointer_list));
      p = p->next;

      p->pointer = host_ptr;
      p->next = NULL;
      
      return host_ptr;
    }
  else
    return malloc (size);
}

void
pocl_pthread_free (void *data, void *ptr)
{
  struct data *d;
  struct pointer_list *p;

  d = (struct data *) data;

  p = d->host_buffers;
  while (p != NULL)
    {
      if (p->pointer == ptr)
        return;

      p = p->next;
    }
  
  free (ptr);
}

void
pocl_pthread_read (void *data, void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
}

//#define DEBUG_MT
//#define DEBUG_MAX_THREAD_COUNT
/**
 * Return an estimate for the maximum thread count that should produce
 * the maximum parallelism without extra threading overheads.
 */
int 
get_max_thread_count() {
  /* query from /proc/cpuinfo how many hardware threads there are, if available */
  const char* cpuinfo = "/proc/cpuinfo";
  /* eight is a good round number ;) */
  const int FALLBACK_MAX_THREAD_COUNT = 8;
  if (access (cpuinfo, R_OK) == 0) 
    {
      FILE *f = fopen (cpuinfo, "r");
#     define MAX_CPUINFO_SIZE 4096
      char contents[MAX_CPUINFO_SIZE];
      int num_read = fread (contents, 1, MAX_CPUINFO_SIZE - 1, f);            
      fclose (f);
      contents[num_read] = '\0';

      /* Count the number of times 'processor' keyword is found which
         should give the number of cores overall in a multiprocessor
         system. In Meego Harmattan on ARM it prints Processor instead of
         processor */
      int cores = 0;
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
  workgroup w;

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
			LLC " -relocation-model=dynamic-no-pic -o %s %s",
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
			"clang -c -o %s.o %s",
			module,
			assembly);
      assert (error >= 0);
      
      error = system (command);
      assert (error == 0);

      error = snprintf (command, COMMAND_LENGTH,
                       "ld -X " SHARED " -o %s %s.o",
                       module,
                       module);
      assert (error >= 0);

      error = system (command);
      assert (error == 0);
      
      d->current_dlhandle = lt_dlopen (module);
      assert (d->current_dlhandle != NULL);

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
  
  w = (workgroup) lt_dlsym (d->current_dlhandle, workgroup_string);
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

    pc->group_id[0] = first_gid_x;

    arguments[i].data = data;
    arguments[i].kernel = kernel;
    arguments[i].device = device;
    arguments[i].pc = *pc;
    arguments[i].workgroup = w;
    arguments[i].last_gid_x = last_gid_x;

    pthread_create (&threads[i],
                    NULL,
                    workgroup_thread,
                    &arguments[i]);
  }

  for (i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
#ifdef DEBUG_MT       
    printf("### thread %x finished\n", (unsigned)threads[i]);
#endif
  }

  free(threads);
  free(arguments);
}

void *
workgroup_thread (void *p)
{
  struct thread_arguments *ta = (struct thread_arguments *) p;
  void *arguments[ta->kernel->num_args];
  struct pocl_argument_list *al;  
  unsigned i = 0;
  al = ta->kernel->arguments;
  while (al != NULL)
    {
      if (ta->kernel->arg_is_local[i])
        {
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_pthread_malloc(ta->data, 0, al->size, NULL);
        } else if (ta->kernel->arg_is_pointer[i])
        arguments[i] = &((*(cl_mem *) (al->value))->device_ptrs[ta->device]);
      else
        arguments[i] = al->value;
      
      ++i;
      al = al->next;
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
  
  i = 0;
  al = ta->kernel->arguments;
  while (al != NULL)
    {
      if (ta->kernel->arg_is_local[i])
        {
          pocl_pthread_free(ta->data, *(void **)(arguments[i]));
          free (arguments[i]);
        }

      ++i;
      al = al->next;
    }
  return NULL;
}
