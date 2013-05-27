/* OpenCL runtime library: clCreateProgramWithBinary()

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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
#include "pocl_util.h"
#include <string.h>

CL_API_ENTRY cl_program CL_API_CALL
POname(clCreateProgramWithBinary)(cl_context                     context,
                          cl_uint                        num_devices,
                          const cl_device_id *           device_list,
                          const size_t *                 lengths,
                          const unsigned char **         binaries,
                          cl_int *                       binary_status,
                          cl_int *                       errcode_ret)
  CL_API_SUFFIX__VERSION_1_0
{
  cl_program program;
  unsigned total_binary_size;
  unsigned char *pos;
  int i;
  int j;
  int errcode;

  if (device_list == NULL || num_devices == 0 || lengths == NULL)
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR;
  }

  if (context == NULL)
  {
    errcode = CL_INVALID_CONTEXT;
    goto ERROR;
  }

  total_binary_size = 0;
  for (i = 0; i < num_devices; ++i)
    {
      if (lengths[i] == 0 || binaries[i] == NULL)
      {
        errcode = CL_INVALID_VALUE;
        goto ERROR;
      }
      total_binary_size += lengths[i];
    }

  // check for invalid devices in device_list[].
  for (i = 0; i < num_devices; i++)
    {
      int found = 0;
      for (j = 0; j < context->num_devices; j++)
        {
          found |= context->devices[i] == device_list[i];
        }
      if (!found)
        {
          errcode = CL_INVALID_DEVICE;
          goto ERROR;
        }
    }
  
  // check for duplicates in device_list[].
  for (i = 0; i < context->num_devices; i++)
    {
      int count = 0;
      for (j = 0; j < num_devices; j++)
        {
          count += context->devices[i] == device_list[j];
        }
      // duplicate devices
      if (count > 1)
        {
          errcode = CL_INVALID_DEVICE;
          goto ERROR;
        }
    }
  
  if ((program = (cl_program) malloc (sizeof (struct _cl_program))) == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }
  
  POCL_INIT_OBJECT(program);
  program->binary_sizes = NULL;
  program->binaries = NULL;
  program->compiler_options = NULL;

  /* Allocate a continuous chunk of memory for all the binaries. */
  if ((program->binary_sizes = 
       (size_t*) malloc (sizeof (size_t) * num_devices)) == NULL ||
      (program->binaries = (unsigned char**) 
       malloc (sizeof (unsigned char*) * num_devices)) == NULL ||
      (program->binaries[0] = (unsigned char*)
       malloc (sizeof (unsigned char) * total_binary_size)) == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN_PROGRAM_AND_BINARIES;
    }

  program->context = context;
  program->num_devices = num_devices;
  program->devices = malloc (sizeof(cl_device_id) * num_devices);
  program->source = NULL;
  program->kernels = NULL;
  /* Create the temporary directory where all kernel files and compilation
     (intermediate) results are stored. */
  program->temp_dir = pocl_create_temp_dir();

  pos = program->binaries[0];
  for (i = 0; i < num_devices; ++i)
    {
      program->devices[i] = device_list[i];
      program->binary_sizes[i] = lengths[i];
      memcpy (pos, binaries[i], lengths[i]);
      program->binaries[i] = pos;
      pos += lengths[i];
      if (binary_status != NULL) /* TODO: validate the binary */
          binary_status[i] = CL_SUCCESS;
    }
  POCL_RETAIN_OBJECT(context);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return program;

ERROR_CLEAN_PROGRAM_BINARIES_AND_DEVICES:
  free(program->devices);
ERROR_CLEAN_PROGRAM_AND_BINARIES:
  free(program->binaries[0]);
  free(program->binaries);
  free(program->binary_sizes);
ERROR_CLEAN_PROGRAM:
  free(program);
ERROR:
    if(errcode_ret != NULL)
    {
        *errcode_ret = errcode;
    }
    return NULL;
}
POsym(clCreateProgramWithBinary)
