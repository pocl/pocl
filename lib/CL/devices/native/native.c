/* OpenCL native device implementation.

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

#include "native.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128

#define ALIGNMENT (max(ALIGNOF_FLOAT16, ALIGNOF_DOUBLE16))

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
};

size_t pocl_native_max_work_item_sizes[] = {1};

void
pocl_native_init (cl_device_id device)
{
  struct data *d;
  
  d = (struct data *) malloc (sizeof (struct data));
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;
}

void *
pocl_native_malloc (void *data, cl_mem_flags flags,
		    size_t size, void *host_ptr)
{
  void *b;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      if (posix_memalign (&b, ALIGNMENT, size) == 0)
	{
	  memcpy (b, host_ptr, size);
	  return b;
	}

      return NULL;
    }

  if (host_ptr != NULL)
    {
      return host_ptr;
    }

  if (posix_memalign (&b, ALIGNMENT, size) == 0)
    return b;

  return NULL;
}

void
pocl_native_free (void *data, cl_mem_flags flags, void *ptr)
{
  if (flags & CL_MEM_COPY_HOST_PTR)
    return;
  
  free (ptr);
}

void
pocl_native_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
}

void
pocl_native_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (device_ptr, host_ptr, cb);
}

void
pocl_native_run (void *data, const char *parallel_filename,
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
  struct pocl_argument *p;
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
  
  w = (pocl_workgroup) lt_dlsym (d->current_dlhandle, workgroup_string);
  assert (w != NULL);

  void *arguments[kernel->num_args];

  for (z = 0; z < pc->num_groups[2]; ++z)
    {
      for (y = 0; y < pc->num_groups[1]; ++y)
	{
	  for (x = 0; x < pc->num_groups[0]; ++x)
	    {
              for (i = 0; i < kernel->num_args; ++i)
                {
                  p = &(kernel->arguments[i]);
		  if (kernel->arg_is_local[i])
		    {
		      arguments[i] = malloc (sizeof (void *));
		      *(void **)(arguments[i]) = pocl_native_malloc(data, 0, p->size, NULL);
		    }
		  else if (kernel->arg_is_pointer[i])
		    arguments[i] = &((*(cl_mem *) (p->value))->device_ptrs[device]);
		  else
		    arguments[i] = p->value;
                }
              for (i = kernel->num_args;
                   i < kernel->num_args + kernel->num_locals;
                   ++i)
                {
                  p = &(kernel->arguments[i]);
                  arguments[i] = malloc (sizeof (void *));
                  *(void **)(arguments[i]) = pocl_native_malloc(data, 0, p->size, NULL);
                }

	      pc->group_id[0] = x;
	      pc->group_id[1] = y;
	      pc->group_id[2] = z;

	      w (arguments, pc);

              for (i = 0; i < kernel->num_args; ++i)
                {
		  if (kernel->arg_is_local[i])
                    pocl_native_free(data, 0, *(void **)(arguments[i]));
                }
              for (i = kernel->num_args;
                   i < kernel->num_args + kernel->num_locals;
                   ++i)
                pocl_native_free(data, 0, *(void **)(arguments[i]));
	    }
	}
    }
}
