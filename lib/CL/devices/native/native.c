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

#define COMMAND_LENGTH 256
#define ARGUMENT_STRING_LENGTH 32

size_t locl_native_max_work_item_sizes[] = {1};

void *
locl_native_malloc(void *data, cl_mem_flags flags,
		   size_t size, void *host_ptr)
{
  if (host_ptr != NULL)
    return host_ptr;
  else
    return malloc(size);
}

void
locl_native_free(void *data, void *ptr)
{
}

void
locl_native_read(void *data, void *host_ptr, void *device_ptr)
{
}

void
locl_native_run(void *data, const char *parallel_filename,
		cl_kernel kernel,
		size_t x, size_t y, size_t z)
{
  char template[] = ".naruXXXXXX";
  char *tmpdir;
  int error;
  char bytecode[LOCL_FILENAME_LENGTH];
  char assembly[LOCL_FILENAME_LENGTH];
  char module[LOCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];
  lt_dlhandle dlhandle;
  char arg_string[ARGUMENT_STRING_LENGTH];
  char size_string[ARGUMENT_STRING_LENGTH];
  void *native_arg, *kernel_arg;
  size_t *size;
  unsigned i;
  workgroup w;

  tmpdir = mkdtemp(template);
  assert(tmpdir != NULL);

  error = snprintf(bytecode, LOCL_FILENAME_LENGTH,
		   "%s/parallel.bc",
		   tmpdir);
  assert(error >= 0);

  error = snprintf(command, COMMAND_LENGTH,
		   LLVM_LD " -link-as-library -o %s %s",
		   bytecode,
		   parallel_filename);
  assert(error >= 0);

  error = system(command);
  assert(error == 0);

  error = snprintf(assembly, LOCL_FILENAME_LENGTH,
		   "%s/parallel.s",
		   tmpdir);
  assert(error >= 0);
  
  error = snprintf(command, COMMAND_LENGTH,
		   LLC " -relocation-model=pic -o %s %s",
		   assembly,
		   bytecode);
  assert(error >= 0);

  error = system(command);
  assert(error == 0);

  error = snprintf(module, LOCL_FILENAME_LENGTH,
		   "%s/parallel.so",
		   tmpdir);
  assert(error >= 0);
  
  error = snprintf(command, COMMAND_LENGTH,
		   "gcc -bundle -o %s %s",
		   module,
		   assembly);
  assert(error >= 0);

  error = system(command);
  assert(error == 0);

  dlhandle = lt_dlopen(module);
  assert(dlhandle != NULL);


/*   kernel_arguments = (struct locl_argument *) lt_dlsym(kernel->dlhandle, "_arguments"); */
/*   native_arguments = (struct locl_argument *) lt_dlsym(dlhandle, "_arguments"); */
/*   assert(kernel_arguments != NULL && native_arguments != NULL); */

  for (i = 0; i < kernel->num_args; ++i) {
    error = snprintf(arg_string, ARGUMENT_STRING_LENGTH,
		     "_arg%d", i);
    assert(error > 0);
    error = snprintf(size_string, ARGUMENT_STRING_LENGTH,
		     "_size%d", i);
    assert(error > 0);

    native_arg = lt_dlsym(dlhandle, arg_string);
    kernel_arg = lt_dlsym(kernel->dlhandle, arg_string);
    size = (size_t *) lt_dlsym(kernel->dlhandle, size_string);
    assert((native_arg != NULL) && (kernel_arg != NULL) && (size != NULL));

    memcpy(native_arg, kernel_arg, *size);
  }

  w = (workgroup) lt_dlsym(dlhandle, "_workgroup");
  assert (w != NULL);

  w(x, y, z);
}
