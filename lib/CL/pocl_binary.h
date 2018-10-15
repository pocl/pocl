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


/* check if buffer is pocl binary format, returns 1 if true */
int pocl_binary_check_binary(cl_device_id device, const unsigned char *binary);



/* returns the size of pocl_binaries[device_i] for allocation */
size_t pocl_binary_sizeof_binary(cl_program program, unsigned device_i);

/* unpacks the content of program->pocl_binaries[device_i] into pocl cache */
cl_int pocl_binary_deserialize(cl_program program, unsigned device_i);

/* pocl cache -> program->pocl_binaries[device_i] */
cl_int pocl_binary_serialize(cl_program program, unsigned device_i, size_t *size);


/* sets the program's build_hash[device_i] for creating a program directory */
void pocl_binary_set_program_buildhash(cl_program program,
                                       unsigned device_i,
                                       const unsigned char *binary);

/* returns the number of kernels without unpacking the binary */
cl_uint pocl_binary_get_kernel_count (cl_program program, unsigned device_i);

/* sets up cl_kernel's metadata, without unpacking the binary in pocl kcache */
cl_int pocl_binary_get_kernels_metadata (cl_program program,
                                         unsigned device_i);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
