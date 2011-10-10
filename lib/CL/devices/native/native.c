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
#define ARGUMENT_STRING_LENGTH 32

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

size_t locl_native_max_work_item_sizes[] = {1};

void
locl_native_init (cl_device_id device)
{
  struct data *d;
  
  d = (struct data *) malloc (sizeof (struct data));
  
  d->host_buffers = NULL;
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;
}

void *
locl_native_malloc (void *data, cl_mem_flags flags,
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
locl_native_free (void *data, void *ptr)
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
locl_native_read (void *data, void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
}

void
locl_native_run (void *data, const char *parallel_filename,
		 struct locl_argument_list *arguments,
		 cl_kernel kernel,
		 size_t x, size_t y, size_t z)
{
  struct data *d;
  char template[] = ".naruXXXXXX";
  char *tmpdir;
  int error;
  char bytecode[LOCL_FILENAME_LENGTH];
  char assembly[LOCL_FILENAME_LENGTH];
  char module[LOCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];
  char arg_string[ARGUMENT_STRING_LENGTH];
  void *arg;
  struct locl_argument_list *p;
  unsigned i;
  workgroup w;

  d = (struct data *) data;

  if (d->current_kernel != kernel)
    {
      tmpdir = mkdtemp (template);
      assert (tmpdir != NULL);
      
      error = snprintf (bytecode, LOCL_FILENAME_LENGTH,
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
      
      error = snprintf (assembly, LOCL_FILENAME_LENGTH,
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
      
      error = snprintf (module, LOCL_FILENAME_LENGTH,
			"%s/parallel.so",
			tmpdir);
      assert (error >= 0);
      
      error = snprintf (command, COMMAND_LENGTH,
			"gcc " SHARED " -o %s %s",
			module,
			assembly);
      assert (error >= 0);
      
      error = system (command);
      assert (error == 0);
      
      d->current_dlhandle = lt_dlopen (module);
      assert (d->current_dlhandle != NULL);

      d->current_kernel = kernel;
    }

  i = 0;
  p = arguments;
  while (p != NULL)
    {
      error = snprintf (arg_string, ARGUMENT_STRING_LENGTH,
			"_arg%d", i);
      assert (error > 0);
      
      arg = lt_dlsym (d->current_dlhandle, arg_string);

      memcpy (arg, p->value, p->size);

      ++i;
      p = p->next;
    }

  w = (workgroup) lt_dlsym (d->current_dlhandle, "_workgroup");
  assert (w != NULL);

  w (x, y, z);
}
