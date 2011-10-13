/* OpenCL runtime library: clEnqueueNDRangeKernel()

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

#include "locl_cl.h"
#include <sys/stat.h>
#include <unistd.h>

#define COMMAND_LENGTH 256
#define ARGUMENT_STRING_LENGTH 32

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t *global_work_offset,
                       const size_t *global_work_size,
                       const size_t *local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  char template[] = ".clekXXXXXX";
  size_t offset_x, offset_y, offset_z;
  size_t global_x, global_y, global_z;
  size_t local_x, local_y, local_z;
  char *tmpdir;
  char kernel_filename[LOCL_FILENAME_LENGTH];
  FILE *kernel_file;
  char parallel_filename[LOCL_FILENAME_LENGTH];
  size_t x, y, z;
  size_t n;
  struct stat buf;
  char command[COMMAND_LENGTH];
  int error;
  struct locl_argument_list *p;
  unsigned i;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;
  
  if (kernel == NULL)
    return CL_INVALID_KERNEL;

  if (command_queue->context != kernel->context)
    return CL_INVALID_CONTEXT;

  if (work_dim < 1 || work_dim > 3)
    return CL_INVALID_WORK_DIMENSION;

  if (global_work_offset != NULL)
    {
      offset_x = global_work_offset[0];
      offset_y = work_dim > 1 ? global_work_offset[1] : 1;
      offset_z = work_dim > 2 ? global_work_offset[2] : 1;
    }
  else
    {
      offset_x = 0;
      offset_y = 0;
      offset_z = 0;
    }
    
  global_x = global_work_size[0];
  global_y = work_dim > 1 ? global_work_size[1] : 1;
  global_z = work_dim > 2 ? global_work_size[2] : 1;

  local_x = local_work_size[0];
  local_y = work_dim > 1 ? local_work_size[1] : 1;
  local_z = work_dim > 2 ? local_work_size[2] : 1;

  tmpdir = mkdtemp(template);
  if (tmpdir == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  error = snprintf(kernel_filename, LOCL_FILENAME_LENGTH,
		   "%s/kernel.bc",
		   tmpdir);
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;

  kernel_file = fopen(kernel_filename, "w+");
  if (kernel_file == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  n = fwrite(kernel->program->binary, 1,
	     kernel->program->binary_size, kernel_file);
  if (n < kernel->program->binary_size)
    return CL_OUT_OF_HOST_MEMORY;
  
  fclose(kernel_file);

  error = snprintf(parallel_filename, LOCL_FILENAME_LENGTH,
		   "%s/parallel.bc",
		   tmpdir);
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;
 
  if (stat(BUILDDIR "/scripts/" LOCL_WORKGROUP, &buf) == 0)
    error = snprintf(command, COMMAND_LENGTH,
		     BUILDDIR "/scripts/" LOCL_WORKGROUP " -k %s -x %zu -y %zu -z %zu -o %s %s",
		     kernel->function_name,
		     local_x, local_y, local_z,
		     parallel_filename, kernel_filename);
  else
    error = snprintf(command, COMMAND_LENGTH,
		     LOCL_WORKGROUP " -k %s -x %zu -y %zu -z %zu -o %s %s",
		     kernel->function_name,
		     local_x, local_y, local_z,
		     parallel_filename, kernel_filename);
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;

  error = system(command);
  if (error != 0)
    return CL_OUT_OF_RESOURCES;
  
  p = kernel->arguments;
  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_is_local[i])
	{
	  if (p->value == NULL)
	    p->value = malloc (sizeof (void *));
	  
	  *(void **)(p->value) = command_queue->device->malloc(command_queue->device->data,
							       0,
							       p->size,
							       NULL);
	  p->size = sizeof (void *);
	}

      p = p->next;
    }

  for (z = 0; z < global_z / local_z; ++z)
    {
      for (y = 0; y < global_y / local_y; ++y)
	{
	  for (x = 0; x < global_x / local_x; ++x)
	    command_queue->device->run(command_queue->device->data,
				       parallel_filename,
				       kernel,
				       x, y, z);
	}
    }

  p = kernel->arguments;
  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_is_local[i])
	command_queue->device->free (command_queue->device->data,
				     *(void**)(p->value));
    }

  return CL_SUCCESS;
}
