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

#include "pocl_binary.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_shared.h"
#include "pocl_util.h"
#include <string.h>

/* creates either a program with binaries, or an empty program. The latter
 * is useful for clLinkProgram() which needs an empty program to put the
 * compiled results in.
 */
cl_program
create_program_skeleton (cl_context context, cl_uint num_devices,
                         const cl_device_id *device_list,
                         const size_t *lengths, const unsigned char **binaries,
                         cl_int *binary_status, cl_int *errcode_ret,
                         int allow_empty_binaries)
{
  cl_program program;
  unsigned i,j;
  int errcode;
  cl_device_id *unique_devlist = NULL;

  POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND((device_list == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((num_devices == 0), CL_INVALID_VALUE);

  if (!allow_empty_binaries)
    {
      POCL_GOTO_ERROR_COND ((lengths == NULL), CL_INVALID_VALUE);

      for (i = 0; i < num_devices; ++i)
        {
          POCL_GOTO_ERROR_ON ((lengths[i] == 0 || binaries[i] == NULL),
                              CL_INVALID_VALUE,
                              "%i-th binary is NULL or its length==0\n", i);
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
      POCL_GOTO_ERROR_ON((count > 1), CL_INVALID_DEVICE,
        "device %s specified multiple times\n", context->devices[i]->long_name);
    }

  // convert subdevices to devices and remove duplicates
  cl_uint real_num_devices = 0;
  unique_devlist = pocl_unique_device_list(device_list, num_devices, &real_num_devices);
  num_devices = real_num_devices;
  device_list = unique_devlist;

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

  if ((program = (cl_program) calloc (1, sizeof (struct _cl_program))) == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
  
  POCL_INIT_OBJECT(program);

  if ((program->binary_sizes =
       (size_t*) calloc (num_devices, sizeof(size_t))) == NULL ||
      (program->binaries = (unsigned char**)
       calloc (num_devices, sizeof(unsigned char*))) == NULL ||
      (program->pocl_binaries = (unsigned char**)
       calloc (num_devices, sizeof(unsigned char*))) == NULL ||
      (program->pocl_binary_sizes =
             (size_t*) calloc (num_devices, sizeof(size_t))) == NULL ||
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
  program->devices = unique_devlist;
  program->build_status = CL_BUILD_NONE;
  program->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;
  char program_bc_path[POCL_FILENAME_LENGTH];

  if (allow_empty_binaries && (lengths == NULL) && (binaries == NULL))
    goto SUCCESS;

  for (i = 0; i < num_devices; ++i)
    {
#ifdef OCS_AVAILABLE
      /* LLVM IR */
      if (!strncmp((const char *)binaries[i], "BC", 2))
        {
          program->binary_sizes[i] = lengths[i];
          program->binaries[i] = (unsigned char*) malloc(lengths[i]);
          memcpy (program->binaries[i], binaries[i], lengths[i]);
          if (binary_status != NULL)
            binary_status[i] = CL_SUCCESS;
        }
      else
#endif
      /* Poclcc binary */
      if (pocl_binary_check_binary(device_list[i], binaries[i]))
        {
          program->pocl_binary_sizes[i] = lengths[i];
          program->pocl_binaries[i] = (unsigned char*) malloc (lengths[i]);
          memcpy (program->pocl_binaries[i], binaries[i], lengths[i]);

          pocl_binary_set_program_buildhash (program, i, binaries[i]);
          int error = pocl_cache_create_program_cachedir
            (program, i, NULL, 0, program_bc_path);
          POCL_GOTO_ERROR_ON((error != 0), CL_BUILD_PROGRAM_FAILURE,
                             "Could not create program cachedir");
          POCL_GOTO_ERROR_ON(pocl_binary_deserialize (program, i),
                             CL_INVALID_BINARY,
                             "Could not unpack a pocl binary\n");
          /* read program.bc, can be useful later */
          if (pocl_exists (program_bc_path))
            {
              pocl_read_file (program_bc_path,
                              (char **)(&program->binaries[i]),
                              (uint64_t *)(&program->binary_sizes[i]));
            }

          if (binary_status != NULL)
            binary_status[i] = CL_SUCCESS;
        }
      /* Unknown binary */
      else
        {
          POCL_MSG_WARN ("Could not recognize binary\n");
          if (binary_status != NULL)
            binary_status[i] = CL_INVALID_BINARY;
          errcode = CL_INVALID_BINARY;
          goto ERROR_CLEAN_PROGRAM_AND_BINARIES;
        }
    }

SUCCESS:
  POCL_RETAIN_OBJECT(context);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return program;

ERROR_CLEAN_PROGRAM_AND_BINARIES:
  if (program->binaries)
    for (i = 0; i < num_devices; ++i)
      POCL_MEM_FREE(program->binaries[i]);
  POCL_MEM_FREE(program->binaries);
  POCL_MEM_FREE(program->binary_sizes);
  if (program->pocl_binaries)
    for (i = 0; i < num_devices; ++i)
      POCL_MEM_FREE(program->pocl_binaries[i]);
  POCL_MEM_FREE(program->pocl_binaries);
  POCL_MEM_FREE(program->pocl_binary_sizes);
/*ERROR_CLEAN_PROGRAM:*/
  POCL_MEM_FREE(program);
ERROR:
  POCL_MEM_FREE(unique_devlist);
    if(errcode_ret != NULL)
      {
        *errcode_ret = errcode;
      }
    return NULL;
}

CL_API_ENTRY cl_program CL_API_CALL POname (clCreateProgramWithBinary) (
    cl_context context, cl_uint num_devices, const cl_device_id *device_list,
    const size_t *lengths, const unsigned char **binaries,
    cl_int *binary_status, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  return create_program_skeleton (context, num_devices, device_list, lengths,
                                  binaries, binary_status, errcode_ret, 0);
}
POsym(clCreateProgramWithBinary)
