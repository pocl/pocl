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
  unsigned i,j;
  int errcode;

  POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND((device_list == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((num_devices == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((lengths == NULL), CL_INVALID_VALUE);

  for (i = 0; i < num_devices; ++i)
    {
      POCL_GOTO_ERROR_ON((lengths[i] == 0 || binaries[i] == NULL), CL_INVALID_VALUE,
        "%i-th binary is NULL or its length==0\n", i);
    }

  // check for invalid devices in device_list[].
  for (i = 0; i < num_devices; i++)
    {
      int found = 0;
      for (j = 0; j < context->num_devices; j++)
        {
          found |= context->devices[j] == device_list[i];
        }
      POCL_GOTO_ERROR_ON((!found), CL_INVALID_DEVICE,
        "device not found in the device list of the context\n");
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
      POCL_GOTO_ERROR_ON((count > 1), CL_INVALID_DEVICE,
        "device %s specified multiple times\n", context->devices[i]->long_name);
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
  program->llvm_irs = NULL;

  if ((program->binary_sizes =
       (size_t*) calloc (num_devices, sizeof(size_t))) == NULL ||
      (program->binaries = (unsigned char**)
       calloc (num_devices, sizeof(unsigned char*))) == NULL ||
      (program->build_log = (char**)
       calloc (num_devices, sizeof(char*))) == NULL ||
      ((program->llvm_irs =
        (void**) calloc (num_devices, sizeof(void*))) == NULL) ||
      ((program->build_hash = (SHA1_digest_t*)
        calloc (num_devices, sizeof(SHA1_digest_t))) == NULL))
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN_PROGRAM_AND_BINARIES;
    }

  program->context = context;
  program->num_devices = num_devices;
  program->devices = (cl_device_id*) calloc (num_devices, sizeof(cl_device_id));
  program->source = NULL;
  program->kernels = NULL;
  program->build_status = CL_BUILD_NONE;

  for (i = 0; i < num_devices; ++i)
    {
      program->devices[i] = device_list[i];
      program->binary_sizes[i] = lengths[i];
      program->binaries[i] = (unsigned char*) malloc (lengths[i]);
      memcpy (program->binaries[i], binaries[i], lengths[i]);
      if (binary_status != NULL) /* TODO: validate the binary */
        binary_status[i] = CL_SUCCESS;
    }

  POCL_RETAIN_OBJECT(context);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return program;

#if 0
ERROR_CLEAN_PROGRAM_BINARIES_AND_DEVICES:
  POCL_MEM_FREE(program->devices);
#endif
ERROR_CLEAN_PROGRAM_AND_BINARIES:
  if (program->binaries)
    for (i = 0; i < num_devices; ++i)
      POCL_MEM_FREE(program->binaries[i]);
  POCL_MEM_FREE(program->binaries);
  POCL_MEM_FREE(program->binary_sizes);
/*ERROR_CLEAN_PROGRAM:*/
  POCL_MEM_FREE(program);
ERROR:
    if(errcode_ret != NULL)
      {
        *errcode_ret = errcode;
      }
    return NULL;
}
POsym(clCreateProgramWithBinary)
