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
#include <unistd.h>

#define COMMAND_LENGTH 256
#define WORKGROUP_STRING_LENGTH 128

struct pointer_list {
  void *pointer;
  struct pointer_list *next;
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

size_t pocl_native_max_work_item_sizes[] = {1};

void
pocl_native_init (cl_device_id device)
{
  struct data *d;
  
  d = (struct data *) malloc (sizeof (struct data));
  
  d->host_buffers = NULL;
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;
}

void *
pocl_native_malloc (void *data, cl_mem_flags flags,
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
pocl_native_free (void *data, void *ptr)
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
pocl_native_read (void *data, void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
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
  struct pocl_argument_list *p;
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
      
      error = snprintf (command, COMMAND_LENGTH,
			LLC " -relocation-model=dynamic-no-pic -mcpu=i386 -o %s %s",
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
			"clang " SHARED " -o %s %s",
			module,
			assembly);
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

  void *arguments[kernel->num_args];

  i = 0;
  p = kernel->arguments;
  while (p != NULL)
    {
      if ((kernel->arg_is_pointer[i]) && (!kernel->arg_is_local[i]))
	arguments[i] = &((*(cl_mem *) (p->value))->device_ptrs[device]);
      else
	arguments[i] = p->value;

      ++i;
      p = p->next;
    }

  snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
	    "_%s_workgroup", kernel->function_name);

  w = (workgroup) lt_dlsym (d->current_dlhandle, workgroup_string);
  assert (w != NULL);

  w (arguments, pc);
}
