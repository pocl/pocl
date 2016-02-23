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

#ifndef POCL_BINARY_FORMAT_H
#define POCL_BINARY_FORMAT_H

#include "pocl_cl.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* pocl binary identifier */
#define POCLCC_STRING_ID "poclbin"
#define POCLCC_STRING_ID_LENGTH 7
#define POCLCC_VERSION 1  

/* pocl binary structures */
typedef struct pocl_binary_kernel_s {
  uint32_t sizeof_kernel_name;
  uint32_t num_args;
  uint32_t num_locals;
  uint32_t sizeof_binary;
  char *kernel_name;
  struct pocl_argument *dyn_arguments;
  struct pocl_argument_info *arg_info;
  unsigned char *binary;
} pocl_binary_kernel;

typedef struct pocl_binary_s {
  char endian;
  char pocl_id[POCLCC_STRING_ID_LENGTH];
  uint64_t device_id;
  uint32_t version;
  uint32_t num_kernels;
  pocl_binary_kernel *kernels;
} pocl_binary;

/* free internal structures */
void pocl_binary_free_kernel(pocl_binary_kernel *kernel);

/* check binary format validity */
int pocl_binary_check_binary(cl_device_id device, pocl_binary *binary);

/* get size of struct (serialized) */
size_t pocl_binary_sizeof_binary(pocl_binary *binary);

/* serialization/deserialization of pocl binary */
int pocl_binary_serialize_binary(unsigned char *buffer, size_t sizeof_buffer,
                                 pocl_binary *binary);
int pocl_binary_deserialize_binary(pocl_binary *binary,
                                   unsigned char *buffer, size_t sizeof_buffer);

/* initialize cl_kernel data from a pocl_binary_kernel specify 
   with kernel_name and cl_device_id */
cl_int pocl_binary_add_clkernel_data(unsigned char **binaries, int num_devices,
                                   const char *kernel_name, cl_kernel kernel, 
                                   cl_device_id device);

/* look for a binary with kernel_name and cl_device_id */
int pocl_binary_search_kernel_binary(unsigned char **binaries, int num_devices,
                                     cl_device_id device, const char *kernel_name, 
                                     unsigned char **binary, int *binary_size);

/* initialize structures */
void pocl_binary_init_binary(pocl_binary *binary, cl_device_id device, 
                             int num_kernels, pocl_binary_kernel *kernels);
int pocl_binary_init_kernel(pocl_binary_kernel *kernel, 
                             char *kernel_name, int sizeof_kernel_name,
                             unsigned char *binary, int sizeof_binary, 
                             int num_args, int num_locals,
                             struct pocl_argument *dyn_arguments, 
                             struct pocl_argument_info *arg_info);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
