/* OpenCL runtime library: pocl binary

   Copyright (c) 2016 pocl developers

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

#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "pocl_cl.h"
#include "pocl_binary.h"

#if defined(WORDS_BIGENDIAN) && WORDS_BIGENDIAN == 1
  const char host_endian = 1;
#else
  const char host_endian = 0;
#endif

/***********************************************************/

void pocl_binary_free_binary(pocl_binary *binary)
{
  if (binary != NULL)
    {
      if (binary->kernels != NULL)
        {
          unsigned j;
          for (j=0; j<binary->num_kernels; j++)
            pocl_binary_free_kernel(&(binary->kernels[j]));
          POCL_MEM_FREE(binary->kernels);
        }
    }
}

void pocl_binary_free_kernel(pocl_binary_kernel *kernel)
{
  if (kernel != NULL)
    {
      POCL_MEM_FREE(kernel->kernel_name);
      POCL_MEM_FREE(kernel->binary);
      POCL_MEM_FREE(kernel->dyn_arguments);
      POCL_MEM_FREE(kernel->arg_info);
    }
}

/***********************************************************/

int pocl_binary_check_binary_header(pocl_binary *binary)
{
  if (binary->endian != host_endian)
    return 0;
  if (binary->version != POCLCC_VERSION)
    return 0;
  return !strncmp(binary->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
}

#define FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x100000001b3)
uint64_t pocl_binary_get_device_id(cl_device_id device)
{
  //FNV-1A with vendor_id, llvm_target_triplet and llvm_cpu
  uint64_t result = FNV_OFFSET;
  const char *llvm_tt = device->llvm_target_triplet;
  const char *llvm_cpu = device->llvm_cpu;

  result *= FNV_PRIME;
  result ^= device->vendor_id;
  int i, length = strlen(llvm_tt);
  for (i=0; i<length; i++)
    {
      result *= FNV_PRIME;
      result ^= llvm_tt[i];
    }
  length = strlen(llvm_cpu);
  for (i=0; i<length; i++)
    {
      result *= FNV_PRIME;
      result ^= llvm_cpu[i];
    }

  return result;
}

int pocl_binary_check_device_id(cl_device_id device, pocl_binary *binary)
{
  return pocl_binary_get_device_id(device) == binary->device_id;
}

int pocl_binary_check_binary(cl_device_id device, pocl_binary *binary)
{
  return pocl_binary_check_binary_header(binary) 
    && pocl_binary_check_device_id(device, binary);
}

/***********************************************************/

#define BUFFER_STORE(elem, type) \
  *(type*)buffer = elem; \
  buffer += sizeof(type)

#define BUFFER_READ(elem, type) \
  elem = *(type*)buffer; \
  buffer += sizeof(type)

#define BUFFER_STORE_STR2(elem, len)  \
  do {                          \
    BUFFER_STORE(len, uint32_t);  \
    if (len)                    \
      {                         \
        memcpy(buffer, elem, len); \
        buffer += len;          \
      }                         \
  } while (0)

#define BUFFER_READ_STR2(elem, len) \
  do {                        \
    BUFFER_READ(len, uint32_t); \
    if (len)                  \
      {                       \
        elem = malloc(len);   \
        memcpy(elem, buffer, len);  \
        buffer += len;        \
      }                       \
  } while (0)

#define BUFFER_STORE_STR(elem)  \
  do { uint32_t len = strlen(elem);  \
    BUFFER_STORE_STR2(elem, len); } while (0)

#define BUFFER_READ_STR(elem)  \
  do { uint32_t len = 0;  \
    BUFFER_READ_STR2(elem, len); } while (0)

#define ADD_STRLEN(else_b)            \
    if (serialized)                   \
      {                               \
        unsigned char* t = buffer+res;\
        res += *(uint32_t*)t;         \
        res += sizeof(uint32_t);      \
      }                               \
    else                              \
      {                               \
        res += sizeof(uint32_t);      \
        res += else_b;                \
      }

size_t pocl_binary_sizeof_kernel(int serialized, pocl_binary_kernel *kernel)
{
  size_t res = 0;
  unsigned char *buffer = (unsigned char*)kernel;
  uint32_t num_args, num_locals;

  res += sizeof(kernel->num_args);
  res += sizeof(kernel->num_locals);
  if (serialized)
    {
      BUFFER_READ(num_args, uint32_t);
      BUFFER_READ(num_locals, uint32_t);
      buffer = (unsigned char*)kernel;
    }
  else
    {
      num_args = kernel->num_args;
      num_locals = kernel->num_locals;
    }
  /* dyn args -> pocl_argument */
  res += (num_args + num_locals) * sizeof(uint64_t);

  /* arg info */
  unsigned i;
  for (i=0; i < num_args; i++)
    {
      res += sizeof(cl_kernel_arg_access_qualifier);
      res += sizeof(cl_kernel_arg_address_qualifier);
      res += sizeof(cl_kernel_arg_type_qualifier);
      res += sizeof(char);
      res += sizeof(char);
      res += sizeof(uint32_t);
      ADD_STRLEN(strlen(kernel->arg_info[i].name));
      ADD_STRLEN(strlen(kernel->arg_info[i].type_name));
    }

  ADD_STRLEN(kernel->sizeof_binary);
  ADD_STRLEN(kernel->sizeof_kernel_name);

  return res;
}

size_t pocl_binary_sizeof_binary(pocl_binary *binary)
{
  size_t size = 8 + sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t);
  unsigned i;
  for (i=0; i < binary->num_kernels; i++)
    size += pocl_binary_sizeof_kernel(0, &(binary->kernels[i]));

  return size;
}

size_t pocl_binary_sizeof_binary_serialized(unsigned char *binary)
{
  unsigned char *start_of_binary = binary;

  binary += (8 + sizeof(uint64_t) + sizeof(uint32_t));
  uint32_t num_kernels = *(uint32_t*)binary;
  binary += sizeof(uint32_t);

  unsigned i;
  for (i=0; i < num_kernels; i++)
      binary += pocl_binary_sizeof_kernel(1, (pocl_binary_kernel *)binary);

  return binary - start_of_binary;
}

/***********************************************************/

void pocl_binary_serialize_kernel_to_buffer(pocl_binary_kernel *kernel, 
                                            unsigned char **buf)
{
  unsigned char *buffer = *buf;
  unsigned i;

  BUFFER_STORE(kernel->num_args, uint32_t);
  BUFFER_STORE(kernel->num_locals, uint32_t);

  for (i=0; i < (kernel->num_args + kernel->num_locals); i++)
    {
      BUFFER_STORE(kernel->dyn_arguments[i].size, uint64_t);
    }

  for (i=0; i < kernel->num_args; i++)
    {
      pocl_argument_info *ai = &kernel->arg_info[i];
      BUFFER_STORE(ai->access_qualifier, cl_kernel_arg_access_qualifier);
      BUFFER_STORE(ai->address_qualifier, cl_kernel_arg_address_qualifier);
      BUFFER_STORE(ai->type_qualifier, cl_kernel_arg_type_qualifier);
      BUFFER_STORE(ai->is_local, char);
      BUFFER_STORE(ai->is_set, char);
      BUFFER_STORE(ai->type, uint32_t);
      BUFFER_STORE_STR(ai->name);
      BUFFER_STORE_STR(ai->type_name);
    }

  BUFFER_STORE_STR2(kernel->kernel_name, kernel->sizeof_kernel_name);
  BUFFER_STORE_STR2(kernel->binary, kernel->sizeof_binary);

  *buf = buffer;
}

int pocl_binary_deserialize_kernel_from_buffer(unsigned char **buf, 
                                               pocl_binary_kernel *kernel)
{
  unsigned i;
  unsigned char *buffer = *buf;

  memset(kernel, 0, sizeof(pocl_binary_kernel));
  BUFFER_READ(kernel->num_args, uint32_t);
  BUFFER_READ(kernel->num_locals, uint32_t);

  kernel->dyn_arguments = calloc((kernel->num_args + kernel->num_locals),
                                 sizeof(struct pocl_argument));
  if (!kernel->dyn_arguments)
    goto ERROR;

  for (i=0; i < (kernel->num_args + kernel->num_locals); i++)
    {
      BUFFER_READ(kernel->dyn_arguments[i].size, uint64_t);
      kernel->dyn_arguments[i].value = NULL;
    }

  kernel->arg_info = calloc(kernel->num_args, sizeof(struct pocl_argument_info));
  if (!kernel->arg_info)
    goto ERROR;

  for (i=0; i < kernel->num_args; i++)
    {
      pocl_argument_info *ai = &kernel->arg_info[i];
      BUFFER_READ(ai->access_qualifier, cl_kernel_arg_access_qualifier);
      BUFFER_READ(ai->address_qualifier, cl_kernel_arg_address_qualifier);
      BUFFER_READ(ai->type_qualifier, cl_kernel_arg_type_qualifier);
      BUFFER_READ(ai->is_local, char);
      BUFFER_READ(ai->is_set, char);
      BUFFER_READ(ai->type, uint32_t);
      BUFFER_READ_STR(ai->name);
      BUFFER_READ_STR(ai->type_name);
    }

  BUFFER_READ_STR2(kernel->kernel_name, kernel->sizeof_kernel_name);
  BUFFER_READ_STR2(kernel->binary, kernel->sizeof_binary);

  *buf = buffer;
  return CL_SUCCESS;

ERROR:
  pocl_binary_free_kernel(kernel);
  return CL_OUT_OF_HOST_MEMORY;
}

/***********************************************************/

int pocl_binary_serialize_binary(unsigned char *buffer, size_t sizeof_buffer,
                                 pocl_binary *binary)
{
  unsigned char *end_of_buffer = buffer + sizeof_buffer;

  assert(pocl_binary_check_binary_header(binary));

  BUFFER_STORE(binary->endian, char);
  memcpy(buffer, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  buffer += POCLCC_STRING_ID_LENGTH;
  BUFFER_STORE(binary->device_id, uint64_t);
  BUFFER_STORE(binary->version, uint32_t);
  BUFFER_STORE(binary->num_kernels, uint32_t);

  assert(buffer < end_of_buffer);

  unsigned i;
  for (i=0; i<binary->num_kernels; i++)
    {
      pocl_binary_kernel *kernel = &(binary->kernels[i]);
      pocl_binary_serialize_kernel_to_buffer(kernel, &buffer);
      assert(buffer <= end_of_buffer);
    }
  
  return CL_SUCCESS;
}

int pocl_binary_deserialize_binary(pocl_binary *binary, 
                                   unsigned char *buffer, size_t sizeof_buffer)
{
  unsigned char *end_of_buffer = buffer + sizeof_buffer;

  memset(binary, 0, sizeof(pocl_binary));
  BUFFER_READ(binary->endian, char);
  memcpy(binary->pocl_id, buffer, POCLCC_STRING_ID_LENGTH);
  buffer += POCLCC_STRING_ID_LENGTH;
  BUFFER_READ(binary->device_id, uint64_t);
  BUFFER_READ(binary->version, uint32_t);
  BUFFER_READ(binary->num_kernels, uint32_t);

  assert(pocl_binary_check_binary_header(binary));
  assert(buffer < end_of_buffer);

  if ((binary->kernels = calloc(binary->num_kernels, sizeof(pocl_binary_kernel))) == NULL)
    goto ERROR;

  unsigned i;
  for (i=0; i<binary->num_kernels; i++)
    {
      pocl_binary_kernel *kernel = &(binary->kernels[i]);
      if (pocl_binary_deserialize_kernel_from_buffer(&buffer, kernel) != CL_SUCCESS)
        goto ERROR;
      assert(buffer <= end_of_buffer);
    }
  return CL_SUCCESS;
  
ERROR:
  pocl_binary_free_binary(binary);
  return CL_OUT_OF_HOST_MEMORY;
}

/***********************************************************/

int pocl_binary_search_kernel(unsigned char **binaries, 
                              int num_devices, cl_device_id device, 
                              const char *kernel_name,
                              pocl_binary_kernel **kernel)
{
  cl_int errcode = CL_INVALID_PROGRAM_EXECUTABLE;
  int i;
  pocl_binary binary_pocl;
  for (i=0; i<num_devices; i++)
    {
      POCL_GOTO_ERROR_COND(
        (errcode = pocl_binary_deserialize_binary(
          &binary_pocl, 
          binaries[i], 
          pocl_binary_sizeof_binary_serialized(binaries[i]))) 
        != CL_SUCCESS,
        errcode);
      if (pocl_binary_check_binary(device, &binary_pocl))
        {
          unsigned j;
          for (j=0; j<binary_pocl.num_kernels; j++)
            {
              pocl_binary_kernel *kernel_pocl = &(binary_pocl.kernels[i]);
              if (!strncmp(kernel_pocl->kernel_name, kernel_name, kernel_pocl->sizeof_kernel_name))
                {
                  POCL_GOTO_ERROR_COND(
                    (*kernel = malloc(sizeof(pocl_binary_kernel))) == NULL,
                    CL_OUT_OF_HOST_MEMORY);
                  pocl_binary_init_kernel(
                    *kernel,
                    kernel_pocl->kernel_name,
                    kernel_pocl->sizeof_kernel_name,
                    kernel_pocl->binary,
                    kernel_pocl->sizeof_binary,
                    kernel_pocl->num_args,
                    kernel_pocl->num_locals,
                    kernel_pocl->dyn_arguments,
                    kernel_pocl->arg_info);
                  pocl_binary_free_binary(&binary_pocl);
                  return CL_SUCCESS;
                }
            }
        }
      pocl_binary_free_binary(&binary_pocl);
    }
  return errcode;
ERROR:
  pocl_binary_free_binary(&binary_pocl);
  return errcode;
}

cl_int pocl_binary_add_clkernel_data(unsigned char **binaries, int num_devices, 
                                   const char *kernel_name, cl_kernel kernel, 
                                   cl_device_id device)
{
  cl_int errcode = CL_SUCCESS;
  pocl_binary_kernel *kernel_pocl; 
  POCL_RETURN_ERROR_COND(
    (errcode = pocl_binary_search_kernel(binaries, num_devices, device, 
                                         kernel_name, &kernel_pocl))
    != CL_SUCCESS,
    errcode);

  POCL_RETURN_ERROR_COND(kernel_pocl == NULL, CL_INVALID_PROGRAM_EXECUTABLE);

  kernel->num_args = kernel_pocl->num_args;
  kernel->num_locals = kernel_pocl->num_locals;

  POCL_RETURN_ERROR_COND((kernel->dyn_arguments =
                          calloc((kernel->num_args + kernel->num_locals),
                                  sizeof(struct pocl_argument))
                          ) == NULL,  CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->dyn_arguments, kernel_pocl->dyn_arguments,
         (kernel->num_args + kernel->num_locals) * sizeof(struct pocl_argument));

  POCL_GOTO_ERROR_COND((kernel->arg_info =
                        calloc(kernel->num_args,
                               sizeof(struct pocl_argument_info))
                        ) == NULL,  CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->arg_info, kernel_pocl->arg_info,
         kernel->num_args * sizeof(struct pocl_argument_info));

  POCL_GOTO_ERROR_COND((kernel->reqd_wg_size = malloc(3*sizeof(int))) == NULL,
                       CL_OUT_OF_HOST_MEMORY);

  pocl_binary_free_kernel(kernel_pocl);

  return CL_SUCCESS;
ERROR:
  POCL_MEM_FREE(kernel->reqd_wg_size);
  POCL_MEM_FREE(kernel->dyn_arguments);
  POCL_MEM_FREE(kernel->dyn_arguments);
  POCL_MEM_FREE(kernel->arg_info);
  return errcode;
}

int pocl_binary_search_kernel_binary(unsigned char **binaries, int num_devices, 
                                     cl_device_id device, const char *kernel_name, 
                                     unsigned char **binary, int *binary_size)
{
  cl_int errcode = CL_SUCCESS;

  pocl_binary_kernel *kernel;
  POCL_RETURN_ERROR_COND(
    (errcode = pocl_binary_search_kernel(binaries, num_devices, device, 
                                         kernel_name, &kernel))
    != CL_SUCCESS,
    errcode);

  POCL_RETURN_ERROR_COND(kernel == NULL, CL_INVALID_PROGRAM_EXECUTABLE);
    
  int sizeof_binary = kernel->sizeof_binary;
  POCL_RETURN_ERROR_COND((*binary = malloc(sizeof_binary)) == NULL,
                         CL_OUT_OF_HOST_MEMORY);

  memcpy(*binary, kernel->binary, sizeof_binary);
  *binary_size = kernel->sizeof_binary;

  pocl_binary_free_kernel(kernel);

  return errcode;
}

/***********************************************************/

void pocl_binary_init_binary(pocl_binary *binary, cl_device_id device, 
                             int num_kernels, pocl_binary_kernel *kernels)
{
  binary->endian = host_endian;
  strncpy(binary->pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  binary->version = POCLCC_VERSION;
  binary->device_id = pocl_binary_get_device_id(device);
  binary->num_kernels = num_kernels;
  binary->kernels = kernels;
}

int pocl_binary_init_kernel(pocl_binary_kernel *kernel, 
                            char *kernel_name, int sizeof_kernel_name, 
                            unsigned char *binary, int sizeof_binary, 
                            int num_args, int num_locals,
                            struct pocl_argument *dyn_arguments, 
                            struct pocl_argument_info *arg_info)
{
  cl_int errcode = CL_SUCCESS;
  POCL_GOTO_ERROR_COND(
    (kernel->kernel_name = malloc(sizeof_kernel_name)) 
    == NULL,
    CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->kernel_name, kernel_name, sizeof_kernel_name);
    
  POCL_GOTO_ERROR_COND(
    (kernel->dyn_arguments = calloc((num_args+num_locals), sizeof(struct pocl_argument)))
    == NULL,
    CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->dyn_arguments, dyn_arguments, 
         (num_args+num_locals)*sizeof(struct pocl_argument));
  
  POCL_GOTO_ERROR_COND(
    (kernel->arg_info = calloc((num_args), sizeof(struct pocl_argument_info)))
    == NULL,
    CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->arg_info, arg_info, 
         (num_args)*sizeof(struct pocl_argument_info));

  POCL_GOTO_ERROR_COND(
    (kernel->binary = malloc(sizeof(unsigned char)*sizeof_binary)) 
    == NULL,
    CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->binary, binary, sizeof(unsigned char)*sizeof_binary);
    
  kernel->sizeof_kernel_name = sizeof_kernel_name;
  kernel->sizeof_binary = sizeof_binary;
  kernel->num_args = num_args;
  kernel->num_locals = num_locals;

  return errcode;
ERROR:
  pocl_binary_free_kernel(kernel);
  return errcode;
}
