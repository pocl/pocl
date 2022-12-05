/* OpenCL runtime library: clCreateProgramWithSource()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

extern unsigned long program_c;

CL_API_ENTRY cl_program CL_API_CALL
POname(clCreateProgramWithSource)(cl_context context,
                          cl_uint count,
                          const char **strings,
                          const size_t *lengths,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_program program = NULL;
  size_t size = 0;
  char *source = NULL;
  unsigned i;
  int errcode;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND((count == 0), CL_INVALID_VALUE);

  program = (cl_program) calloc(1, sizeof(struct _cl_program));
  if (program == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT(program);

  for (i = 0; i < count; ++i)
    {
      POCL_GOTO_ERROR_ON((strings[i] == NULL), CL_INVALID_VALUE,
          "strings[%i] is NULL\n", i);

      if (lengths == NULL)
        size += strlen(strings[i]);
      else if (lengths[i] == 0)
        size += strlen(strings[i]);
      else
        size += lengths[i];
    }

  source = (char *) malloc(size + 1);
  if (source == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  program->source = source;

  for (i = 0; i < count; ++i)
    {
      if (lengths == NULL)
        {
          memcpy(source, strings[i], strlen(strings[i]));
          source += strlen(strings[i]);
        }
      else if (lengths[i] == 0)
        {
          memcpy(source, strings[i], strlen(strings[i]));
          source += strlen(strings[i]);
        }
      else
        {
          memcpy(source, strings[i], lengths[i]);
          source += lengths[i];
        }
    }

  *source = '\0';

  program->context = context;
  program->associated_num_devices = context->num_devices;
  program->associated_devices = context->devices;
  program->num_devices = 0;
  program->devices = 0;

  program->build_status = CL_BUILD_NONE;
  program->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;

  if ((program->binary_sizes
       = (size_t *)calloc (program->associated_num_devices, sizeof (size_t)))
          == NULL
      || (program->binaries = (unsigned char **)calloc (
              program->associated_num_devices, sizeof (unsigned char *)))
             == NULL
      || (program->pocl_binaries = (unsigned char **)calloc (
              program->associated_num_devices, sizeof (unsigned char *)))
             == NULL
      || (program->pocl_binary_sizes = (size_t *)calloc (
              program->associated_num_devices, sizeof (size_t)))
             == NULL
      || (program->build_log
          = (char **)calloc (program->associated_num_devices, sizeof (char *)))
             == NULL
      || ((program->data = (void **)calloc (program->associated_num_devices,
                                            sizeof (void *)))
          == NULL)
      || ((program->global_var_total_size = (size_t *)calloc (
               program->associated_num_devices, sizeof (size_t)))
          == NULL)
      || ((program->llvm_irs
           = (void *)calloc (program->associated_num_devices, sizeof (void *)))
          == NULL)
      || ((program->gvar_storage
           = (void *)calloc (program->associated_num_devices, sizeof (void *)))
          == NULL)
      || ((program->build_hash = (SHA1_digest_t *)calloc (
               program->associated_num_devices, sizeof (SHA1_digest_t)))
          == NULL))
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  POname(clRetainContext)(context);

  TP_CREATE_PROGRAM (context->id, program->id);

  POCL_ATOMIC_INC (program_c);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return program;

ERROR:
  if (program) {
    POCL_MEM_FREE(program->build_hash);
    POCL_MEM_FREE (program->data);
    POCL_MEM_FREE (program->global_var_total_size);
    POCL_MEM_FREE (program->llvm_irs);
    POCL_MEM_FREE (program->gvar_storage);
    POCL_MEM_FREE(program->build_log);
    POCL_MEM_FREE(program->binaries);
    POCL_MEM_FREE(program->binary_sizes);
    POCL_MEM_FREE(program->pocl_binaries);
    POCL_MEM_FREE(program->pocl_binary_sizes);
    POCL_MEM_FREE(program->source);
  }
  POCL_MEM_FREE(program);
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym(clCreateProgramWithSource)
