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

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* Returns the cpu name as reported by LLVM. */
char* get_cpu_name();
/* Returns if the cpu supports FMA instruction (uses LLVM). */
int cpu_has_fma();

/* Compiles an .cl file into LLVM IR.
 */
int pocl_llvm_build_program(cl_program program,
                            unsigned device_i,
                            const char* user_options_cstr,
                            char *program_bc_path,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names);


/* Retrieve metadata of the given kernel in the program to populate the
 * cl_kernel object.
 */
int pocl_llvm_get_kernel_metadata
(cl_program program, 
 cl_kernel kernel,
 int device_i,     
 const char* kernel_name,
 int *errcode);

/* This function links the input kernel LLVM bitcode and the
 * OpenCL kernel runtime library into one LLVM module, then
 * runs pocl's kernel compiler passes on that module to produce 
 * a function that executes all work-items in a work-group.
 *
 * Output is a LLVM bitcode file that contains a work-group function
 * and its associated launchers. 
 *
 * TODO: this is not thread-safe, it changes the LLVM global options to
 * control the compilation. We should enforce only one compilations is done
 * at a time or control the options through thread safe methods.
 */
int pocl_llvm_generate_workgroup_function(cl_device_id device,
                                          cl_kernel kernel, size_t local_x,
                                          size_t local_y, size_t local_z);

int pocl_llvm_generate_workgroup_function_nowrite(
    cl_device_id device, cl_kernel kernel, size_t local_x, size_t local_y,
    size_t local_z, void **output);
/**
 * Free the LLVM IR of a program for a given device
 */
void pocl_free_llvm_irs(cl_program program, int device_i);

/* calls delete on the module. */
void pocl_destroy_llvm_module(void *modp);

/**
 * Update the program->binaries[] representation of the kernels
 * from the program->llvm_irs[] representation.
 * Also updates the 'program.bc' file in the POCL_TEMP_DIR cache.
 */
void pocl_llvm_update_binaries (cl_program program);

/**
 * Count the number of "__kernel" functions in 'program'.
 *
 * Results are valid as long as program binary is not modified.
 */
unsigned pocl_llvm_get_kernel_count(cl_program program);

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
unsigned pocl_llvm_get_kernel_names( cl_program program, char **knames, unsigned max_num_krn);

/** Compile the kernel in infile from LLVM bitcode to native object file for
 * device, into outfile.
 */
int pocl_llvm_codegen(cl_kernel kernel, cl_device_id device, void *modp,
                      char **output, size_t *output_size);

/* Parse program file and populate program's llvm_irs */
int
pocl_update_program_llvm_irs(cl_program program, unsigned device_i,
                             cl_device_id device);


int pocl_llvm_link_program(cl_program program,
                           unsigned device_i,
                           char *program_bc_path,
                           cl_uint num_input_programs,
                           unsigned char **cur_device_binaries,
                           size_t *cur_device_binary_sizes,
                           void **cur_llvm_irs);


#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#ifdef __cplusplus
}
#endif
