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

#include "pocl_binary.h"

/***********************************************************/

void pocl_binary_free_binary(pocl_binary *binary)
{
  if (binary != NULL)
    {
      if (binary->kernels != NULL)
        {
          int j;
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

int pocl_binary_sizeof_kernel(pocl_binary_kernel *kernel)
{
  return sizeof(pocl_binary_kernel) 
    + kernel->sizeof_kernel_name + kernel->sizeof_binary
    - sizeof(kernel->kernel_name) - sizeof(kernel->binary) 
    + (kernel->num_args + kernel->num_locals) * sizeof(struct pocl_argument)
    + kernel->num_args * sizeof(struct pocl_argument_info)
    - sizeof (kernel->dyn_arguments) - sizeof(kernel->arg_info);
}

int pocl_binary_sizeof_binary(pocl_binary *binary)
{
  int size = sizeof(pocl_binary) - sizeof(binary->kernels);
  int i;
  for (i=0; i<binary->num_kernels; i++)
    size += pocl_binary_sizeof_kernel(&(binary->kernels[i]));

  return size;
}

int pocl_binary_sizeof_binary_serialized(unsigned char *binary)
{
  pocl_binary *binary_pocl = (pocl_binary *)binary;
  unsigned char *start_of_binary = binary;
  binary += sizeof(pocl_binary) - sizeof(binary_pocl->kernels);
  int num_kernels = binary_pocl->num_kernels;
  int i;
  for (i=0; i<num_kernels; i++)
      binary += pocl_binary_sizeof_kernel((pocl_binary_kernel *)binary);

  return binary - start_of_binary;
}

/***********************************************************/

void pocl_binary_serialize_kernel_to_buffer(pocl_binary_kernel *kernel, 
                                            unsigned char **buf)
{
  unsigned char *buffer = *buf;
  int sizeof_kernel_name = kernel->sizeof_kernel_name;
  int sizeof_binary = kernel->sizeof_binary;
  int sizeof_dyn_args = (kernel->num_args + kernel->num_locals) 
    * sizeof(struct pocl_argument);
  int sizeof_arg_info = kernel->num_args * sizeof(struct pocl_argument_info);

  memcpy(buffer, kernel, sizeof(pocl_binary_kernel));
  buffer = (unsigned char *)(&(((pocl_binary_kernel *)buffer)->kernel_name));

  memcpy(buffer, kernel->kernel_name, sizeof_kernel_name);
  buffer += sizeof_kernel_name;

  memcpy(buffer, kernel->dyn_arguments, sizeof_dyn_args);
  buffer += sizeof_dyn_args;

  memcpy(buffer, kernel->arg_info, sizeof_arg_info);
  buffer += sizeof_arg_info;

  memcpy(buffer, kernel->binary, sizeof_binary);
  buffer += sizeof_binary;
  *buf = buffer;
}

int pocl_binary_deserialize_kernel_from_buffer(unsigned char **buf, 
                                               pocl_binary_kernel *kernel)
{
  unsigned char *kernel_src = *buf;
  pocl_binary_kernel *kernel_tmp_src = (pocl_binary_kernel *)kernel_src;

  memcpy(kernel, kernel_tmp_src, sizeof(pocl_binary_kernel));

  int sizeof_kernel_name = kernel_tmp_src->sizeof_kernel_name;
  int sizeof_binary = kernel_tmp_src->sizeof_binary;
  int sizeof_dyn_args = 
    (kernel_tmp_src->num_args + kernel_tmp_src->num_locals) 
    * sizeof(struct pocl_argument);
  int sizeof_arg_info = 
    kernel_tmp_src->num_args * sizeof(struct pocl_argument_info);

  kernel_src = (unsigned char *)(&(kernel_tmp_src->kernel_name));

  if ((kernel->kernel_name = malloc(sizeof_kernel_name)) == NULL)
    goto ERROR;
  memcpy(kernel->kernel_name, kernel_src, sizeof_kernel_name);
  kernel_src += sizeof_kernel_name;

  if ((kernel->dyn_arguments = malloc(sizeof_dyn_args)) == NULL)
    goto ERROR;
  memcpy(kernel->dyn_arguments, kernel_src, sizeof_dyn_args);
  kernel_src += sizeof_dyn_args;

  if ((kernel->arg_info = malloc(sizeof_arg_info)) == NULL)
    goto ERROR;
  memcpy(kernel->arg_info, kernel_src, sizeof_arg_info);
  kernel_src += sizeof_arg_info;

  if ((kernel->binary = malloc(sizeof_binary)) == NULL)
    goto ERROR;
  memcpy(kernel->binary, kernel_src, sizeof_binary);
  kernel_src += sizeof_binary;

  *buf = kernel_src;
  return CL_SUCCESS;
ERROR:
  pocl_binary_free_kernel(kernel);
  return CL_OUT_OF_HOST_MEMORY;
}

/***********************************************************/

int pocl_binary_serialize_binary(unsigned char *buffer, int sizeof_buffer, 
                                 pocl_binary *binary)
{
  unsigned char *end_of_buffer = buffer + sizeof_buffer;

  assert(pocl_binary_check_binary_header(binary));
  memcpy(buffer, binary, sizeof(pocl_binary));
  buffer = (unsigned char *)(&(((pocl_binary *)buffer)->kernels));
  assert(buffer < end_of_buffer);
      
  int i;
  for (i=0; i<binary->num_kernels; i++)
    {
      pocl_binary_kernel *kernel = &(binary->kernels[i]);
      pocl_binary_serialize_kernel_to_buffer(kernel, &buffer);
      assert(buffer <= end_of_buffer);
    }
  
  return CL_SUCCESS;
}

int pocl_binary_deserialize_binary(pocl_binary *binary, 
                                   unsigned char *buffer, int sizeof_buffer)
{
  unsigned char *end_of_buffer = buffer + sizeof_buffer;

  memcpy(binary, buffer, sizeof(pocl_binary));
  assert(pocl_binary_check_binary_header(binary));
  buffer = (unsigned char *)(&(((pocl_binary *)buffer)->kernels));
  assert(buffer < end_of_buffer);
  
  if ((binary->kernels = malloc(binary->num_kernels*sizeof(pocl_binary_kernel))) == NULL)
    goto ERROR;
      
  int i;
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
          int j;
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

  int sizeof_dyn_args = (kernel->num_args + kernel->num_locals) * sizeof(struct pocl_argument);
  int sizeof_arg_info = kernel->num_args * sizeof(struct pocl_argument_info);

  POCL_RETURN_ERROR_COND((kernel->dyn_arguments = malloc(sizeof_dyn_args)) == NULL,
                         CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->dyn_arguments, kernel_pocl->dyn_arguments, sizeof_dyn_args);

  POCL_GOTO_ERROR_COND((kernel->arg_info = malloc(sizeof_arg_info)) == NULL,
                       CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->arg_info, kernel_pocl->arg_info, sizeof_arg_info);

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
    (kernel->dyn_arguments = malloc((num_args+num_locals)*sizeof(struct pocl_argument))) 
    == NULL,
    CL_OUT_OF_HOST_MEMORY);
  memcpy(kernel->dyn_arguments, dyn_arguments, 
         (num_args+num_locals)*sizeof(struct pocl_argument));
  
  POCL_GOTO_ERROR_COND(
    (kernel->arg_info = malloc((num_args)*sizeof(struct pocl_argument_info))) 
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
