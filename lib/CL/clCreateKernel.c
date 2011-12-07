/* OpenCL runtime library: clCreateKernel()

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

#include "pocl_cl.h"
#include <unistd.h>
#include <sys/stat.h>
#include <unistd.h>

#define COMMAND_LENGTH 1024

CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  char template[] = ".clckXXXXXX";
  cl_kernel kernel;
  char *tmpdir;
  char binary_filename[POCL_FILENAME_LENGTH];
  FILE *binary_file;
  size_t n;
  char descriptor_filename[POCL_FILENAME_LENGTH];
  struct stat buf;
  char command[COMMAND_LENGTH];
  int error;
  lt_dlhandle dlhandle;
  int i;

  if (program == NULL)
    POCL_ERROR(CL_INVALID_PROGRAM);

  if (program->binary == NULL)
    POCL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);

  tmpdir = mkdtemp(template);
  if (tmpdir == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  error = snprintf(binary_filename, POCL_FILENAME_LENGTH,
		   "%s/kernel.bc",
		   tmpdir);
  if (error < 0)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  binary_file = fopen(binary_filename, "w+");
  if (binary_file == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  n = fwrite(program->binary, 1,
	     program->binary_size, binary_file);
  if (n < program->binary_size)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);
  
  fclose(binary_file);

  kernel = (cl_kernel) malloc(sizeof(struct _cl_kernel));
  if (kernel == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  error = snprintf(descriptor_filename, POCL_FILENAME_LENGTH,
		   "%s/descriptor.so",
		   tmpdir);
  if (error < 0)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  if (stat(BUILDDIR "/scripts/" POCL_KERNEL, &buf) == 0)
    error = snprintf(command, COMMAND_LENGTH,
		     BUILDDIR "/scripts/" POCL_KERNEL " -k %s -o %s %s",
		     kernel_name,
		     descriptor_filename,
		     binary_filename);
  else
    error = snprintf(command, COMMAND_LENGTH,
		     POCL_KERNEL " -k %s -o %s %s",
		     kernel_name,
		     descriptor_filename,
		     binary_filename);
  if (error < 0)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  error = system(command);
  if (error != 0)
    POCL_ERROR(CL_INVALID_KERNEL_NAME);

  dlhandle = lt_dlopen(descriptor_filename);
  if (dlhandle == NULL) 
    {
      fprintf(stderr, 
              "Error loading the kernel descriptor from %s (lt_dlerror(): %s)\n", 
              descriptor_filename, lt_dlerror());
      POCL_ERROR(CL_OUT_OF_HOST_MEMORY);
    }

  kernel->function_name = kernel_name;
  kernel->num_args = *(cl_uint *) lt_dlsym(dlhandle, "_num_args");
  kernel->reference_count = 1;
  kernel->context = program->context;
  kernel->program = program;
  kernel->dlhandle = dlhandle;
  kernel->arg_is_pointer = lt_dlsym(dlhandle, "_arg_is_pointer");
  kernel->arg_is_local = lt_dlsym(dlhandle, "_arg_is_local");
  kernel->num_locals = *(cl_uint *) lt_dlsym(dlhandle, "_num_locals");
  kernel->arguments =
    (struct pocl_argument *) malloc ((kernel->num_args + kernel->num_locals) *
                                     sizeof (struct pocl_argument));
  kernel->next = NULL;

  /* Initialize kernel arguments. */
  for (i = 0; i < kernel->num_args; ++i)
    {
      kernel->arguments[i].value = NULL;
      kernel->arguments[i].size = 0;
    }

  /* Fill up automatic local arguments. */
  for (i = 0; i < kernel->num_locals; ++i)
    {
      kernel->arguments[kernel->num_args + i].value = NULL;
      kernel->arguments[kernel->num_args + i].size =
        ((size_t *) lt_dlsym(dlhandle, "_local_size"))[i];
    }

  if (program->kernels == NULL)
    program->kernels = kernel;
  else {
    cl_kernel k = program->kernels;
    while (k->next != NULL)
      k = k->next;
    k->next = kernel;
  }

  return kernel;
}
