/* OpenCL runtime library: clCreateKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012 Pekka Jääskeläinen / Tampere Univ. of Technology
   
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
#include "install-paths.h"
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define COMMAND_LENGTH 1024

CL_API_ENTRY cl_kernel CL_API_CALL
POname(clCreateKernel)(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_kernel kernel;
  char device_tmpdir[POCL_FILENAME_LENGTH];
  char descriptor_filename[POCL_FILENAME_LENGTH];
  int errcode;
  int error;
  lt_dlhandle dlhandle = NULL;
  int i;
  int device_i;
  
  if (program == NULL || program->num_devices == 0)
  {
    errcode = CL_INVALID_PROGRAM;
    goto ERROR;
  }

  if (program->binaries == NULL || program->binary_sizes == NULL)
  {
    errcode = CL_INVALID_PROGRAM_EXECUTABLE;
    goto ERROR;
  }

  kernel = (cl_kernel) malloc(sizeof(struct _cl_kernel));
  if (kernel == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT (kernel);

  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      if (device_i > 0)
        POname(clRetainKernel) (kernel);

      snprintf (device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
                program->temp_dir, program->devices[device_i]->short_name);

      /* If there is no device dir for this device, the program was
         not built for that device in clBuildProgram. This seems to
         be OK by the standard. */
      if (access (device_tmpdir, F_OK) != 0) continue;
 
      error = call_pocl_kernel( program, kernel, device_i, kernel_name, 
                                device_tmpdir, descriptor_filename, &errcode );

      if (error)
        {
          goto ERROR_CLEAN_KERNEL;
        } 

      /* when using the API, there is no descriptor file */
      #ifndef USE_LLVM_API
      if (dlhandle == NULL)
        {
          if (access (descriptor_filename, R_OK) != 0)
            POCL_ABORT("The kernel descriptor.so was not found.\n");
        
          dlhandle = lt_dlopen(descriptor_filename);
          if (dlhandle == NULL) 
            {
              fprintf(stderr, 
                      "Error loading the kernel descriptor from %s (lt_dlerror(): %s)\n", 
                      descriptor_filename, lt_dlerror());
              errcode = CL_OUT_OF_HOST_MEMORY;
              goto ERROR_CLEAN_KERNEL;
            }
        }
      #endif 
    }

  #ifndef USE_LLVM_API
  kernel->num_args = *(cl_uint *) lt_dlsym(dlhandle, "_num_args");
  kernel->reqd_wg_size = (int*)lt_dlsym(dlhandle, "_reqd_wg_size");
  kernel->dlhandle = dlhandle; /* Store this, as the above is a pointer
                                  to the dl image, and it cannot be freed
                                  before the cl_kernel is destroyed */
  kernel->arg_is_pointer = lt_dlsym(dlhandle, "_arg_is_pointer");
  kernel->arg_is_local = lt_dlsym(dlhandle, "_arg_is_local");
  kernel->arg_is_image = lt_dlsym(dlhandle, "_arg_is_image");
  kernel->arg_is_sampler = lt_dlsym(dlhandle, "_arg_is_sampler");
  kernel->num_locals = *(cl_uint *) lt_dlsym(dlhandle, "_num_locals");
  /* Temporary store for the arguments that are set with clSetKernelArg. */
  kernel->dyn_arguments =
    (struct pocl_argument *) malloc ((kernel->num_args + kernel->num_locals) *
                                     sizeof (struct pocl_argument));
  /* Initialize kernel "dynamic" arguments (in case the user doesn't). */
  for (i = 0; i < kernel->num_args; ++i)
    {
      kernel->dyn_arguments[i].value = NULL;
      kernel->dyn_arguments[i].size = 0;
    }

  /* Fill up automatic local arguments. */
  for (i = 0; i < kernel->num_locals; ++i)
    {
      kernel->dyn_arguments[kernel->num_args + i].value = NULL;
      kernel->dyn_arguments[kernel->num_args + i].size =
        ((unsigned *) lt_dlsym(dlhandle, "_local_sizes"))[i];
    }
  #endif

  /* TODO: one of these two could be eliminated?  */
  kernel->function_name = strdup(kernel_name);
  kernel->name = strdup(kernel_name);
  kernel->context = program->context;
  kernel->program = program;
  kernel->next = NULL;

  cl_kernel k = program->kernels;
  program->kernels = kernel;
  kernel->next = k;

  POCL_RETAIN_OBJECT(program);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return kernel;

ERROR_CLEAN_KERNEL_AND_CONTENTS:
  free(kernel->function_name);
  free(kernel->name);
  free(kernel->dyn_arguments);
ERROR_CLEAN_KERNEL:
  free(kernel);
ERROR:
  if(errcode_ret != NULL)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym(clCreateKernel)
