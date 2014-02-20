/* pocl_llvm.h: interface to call LLVM and Clang.

   Copyright (c) 2013 Kalle Raiskila and
                      Pekka Jääskeläinen
   
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

#pragma once
#include "pocl_cl.h"

#ifdef __cplusplus
extern "C" {
#endif

// compile kernel source to bitcode
int call_pocl_build(cl_program program,
                    cl_device_id device,
                    int device_i,     
                    const char* source_file_name,
                    const char* binary_filename,
                    const char* device_tmpdir,
                    const char* user_options );


// create wrapper code for compiling a LLVM IR 
// function as a OpenCL kernel
int call_pocl_kernel(cl_program program, 
                     cl_kernel kernel,
                     int device_i,     
                     const char* kernel_name,
                     const char* device_tmpdir, 
                     char* descriptor_filename,
                     int *errcode );

/* Run the pocl passes on a kernel in LLVM IR, link it with kernel, 
 * and produce the 'paralellized' kernel file.
 */
int call_pocl_workgroup(cl_device_id device,
                        cl_kernel kernel,
                        size_t local_x, size_t local_y, size_t local_z,
                        const char* parallel_filename,
                        const char* kernel_filename );

/**
 * Refresh the on binary representation of the program, update the
 * data in the program object. */
void pocl_llvm_update_binaries (cl_program program);

/**
 * Find the "__kernel" function names in 'program',
 * filling the callee-allocated array with pointer to the program binary.
 * No more than 'max_num_krn' are written.
 *
 * Results are valid as long as program binary is not modified.
 *
 * Returns the number of kernels found in the program (may be greater than
 * 'max_num_krn')
 */
int get_kernel_names( cl_program program, const char **knames, unsigned max_num_krn);

#ifdef __cplusplus
}
#endif

