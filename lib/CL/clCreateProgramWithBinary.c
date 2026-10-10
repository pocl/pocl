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

/* The caller's binaries, lengths and binary_status are indexed by the
 * caller's device list. The program has one slot per distinct root device, in
 * the order pocl_unique_device_list leaves them, so a slot must find its
 * entries by device, not by position: [sub1, sub2, other] reduces to
 * [other, root]. */
static unsigned
caller_index (const cl_device_id *caller_devs, cl_uint caller_num,
              cl_device_id root)
{
  unsigned j;
  for (j = 0; j < caller_num; ++j)
    if (pocl_real_dev (caller_devs[j]) == root)
      return j;
  assert (0 && "every root comes from the caller's list");
  return 0;
}

/* Every entry of the caller's list that reduces to root gets its status. */
static void
set_binary_status (cl_int *binary_status, const cl_device_id *caller_devs,
                   cl_uint caller_num, cl_device_id root, cl_int status)
{
  unsigned j;
  if (binary_status == NULL)
    return;
  for (j = 0; j < caller_num; ++j)
    if (pocl_real_dev (caller_devs[j]) == root)
      binary_status[j] = status;
}

/** Creates either a program with binaries, or an empty program.
 *
 * The latter is useful for clLinkProgram() which needs an empty program to put
 * the compiled results in.
 */
cl_program
create_program_skeleton (cl_context context, cl_uint num_devices,
                         const cl_device_id *device_list,
                         const size_t *lengths, const unsigned char **binaries,
                         cl_int *binary_status, cl_int *errcode_ret,
                         int allow_empty_binaries)
{
  cl_program program = NULL;
  unsigned i,j;
  int errcode = CL_SUCCESS;
  cl_device_id *unique_devlist = NULL;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

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

  // convert subdevices to devices and remove duplicates; the caller's list
  // still indexes binaries[], lengths[] and binary_status[]
  const cl_device_id *caller_devs = device_list;
  cl_uint caller_num = num_devices;
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
      POCL_GOTO_ERROR_ON (
          (*(device_list[i]->available) == CL_FALSE), CL_DEVICE_NOT_AVAILABLE,
          "Requested building for device '%s' but it is unavailable.\n",
          device_list[i]->long_name);
    }

  if ((program = (cl_program) calloc (1, sizeof (struct _cl_program))) == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  POCL_INIT_OBJECT (program, context);

  if ((program->binary_sizes = (size_t *)calloc (num_devices, sizeof (size_t)))
          == NULL
      || (program->binaries
          = (unsigned char **)calloc (num_devices, sizeof (unsigned char *)))
             == NULL
      || (program->pocl_binaries
          = (unsigned char **)calloc (num_devices, sizeof (unsigned char *)))
             == NULL
      || (program->pocl_binary_sizes
          = (size_t *)calloc (num_devices, sizeof (size_t)))
             == NULL
      || (program->build_log = (char **)calloc (num_devices, sizeof (char *)))
             == NULL
      || ((program->data = (void **)calloc (num_devices, sizeof (void *)))
          == NULL)
      || ((program->global_var_total_size
           = (size_t *)calloc (num_devices, sizeof (size_t)))
          == NULL)
      || ((program->llvm_irs
           = (void *)calloc (num_devices, sizeof (void *)))
          == NULL)
      || ((program->gvar_storage
           = (void *)calloc (num_devices, sizeof (void *)))
          == NULL)
      || ((program->build_hash
           = (SHA1_digest_t *)calloc (num_devices, sizeof (SHA1_digest_t)))
          == NULL))
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  program->context = context;
  program->associated_num_devices = num_devices;
  program->associated_devices = unique_devlist;
  program->num_devices = num_devices;
  program->devices = unique_devlist;
  program->build_status = CL_BUILD_NONE;
  program->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;

  TP_CREATE_PROGRAM (context->id, program->id);

  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];

  if (allow_empty_binaries && (lengths == NULL) && (binaries == NULL))
    goto SUCCESS;

  for (i = 0; i < num_devices; ++i)
    {
      unsigned src = caller_index (caller_devs, caller_num, device_list[i]);
      const unsigned char *binary = binaries[src];
      size_t length = lengths[src];

      /* Poclcc binary */
      if (pocl_binary_check_binary (device_list[i], binary))
        {
          program->pocl_binary_sizes[i] = length;
          program->pocl_binaries[i] = (unsigned char *)malloc (length);
          memcpy (program->pocl_binaries[i], binary, length);

          pocl_binary_set_program_buildhash (program, i, binary);
          int error = pocl_cache_create_program_cachedir
            (program, i, NULL, 0, program_bc_path);
          POCL_GOTO_ERROR_ON((error != 0), CL_BUILD_PROGRAM_FAILURE,
                             "Could not create program cachedir");
          POCL_GOTO_ERROR_ON(pocl_binary_deserialize (program, i),
                             CL_INVALID_BINARY,
                             "Could not unpack a pocl binary\n");

          /* read program.bc if present; can be useful later */
          if (pocl_exists (program_bc_path))
            {
              uint64_t size = 0;
              pocl_read_file (program_bc_path,
                              (char **)(&program->binaries[i]), &size);
              program->binary_sizes[i] = (size_t)size;
            }

          set_binary_status (binary_status, caller_devs, caller_num,
                             device_list[i], CL_SUCCESS);
        }
      /* check if the driver supports that binary */
      else
        {
          cl_device_id device = program->associated_devices[i];
          if (device->ops->supports_binary
              && device->ops->supports_binary (device, length,
                                               (const char *)binary))
            {
              program->binary_sizes[i] = length;
              program->binaries[i] = (unsigned char *)malloc (length);
              memcpy (program->binaries[i], binary, length);
              set_binary_status (binary_status, caller_devs, caller_num,
                                 device_list[i], CL_SUCCESS);
            }
          else
            {
              POCL_MSG_WARN ("Could not recognize binary for device %u\n",
                             src);
              set_binary_status (binary_status, caller_devs, caller_num,
                                 device_list[i], CL_INVALID_BINARY);
              errcode = CL_INVALID_BINARY;
              goto ERROR;
            }
        }
    }

SUCCESS:
  POname(clRetainContext)(context);

  POCL_ATOMIC_INC (program_c);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return program;

ERROR:
  if (program)
    {
      if (program->binaries)
        for (i = 0; i < num_devices; ++i)
          POCL_MEM_FREE (program->binaries[i]);
      POCL_MEM_FREE (program->binaries);
      POCL_MEM_FREE (program->binary_sizes);
      if (program->pocl_binaries)
        for (i = 0; i < num_devices; ++i)
          POCL_MEM_FREE (program->pocl_binaries[i]);
      POCL_MEM_FREE (program->pocl_binaries);
      POCL_MEM_FREE (program->pocl_binary_sizes);
      POCL_MEM_FREE (program->data);
      POCL_MEM_FREE (program->global_var_total_size);
      POCL_MEM_FREE (program->llvm_irs);
      POCL_MEM_FREE (program->gvar_storage);
      POCL_MEM_FREE (program->build_log);
      POCL_MEM_FREE (program->build_hash);
      POCL_MEM_FREE (program);
    }
  POCL_MEM_FREE(unique_devlist);
  if (errcode_ret != NULL)
    *errcode_ret = errcode;
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
