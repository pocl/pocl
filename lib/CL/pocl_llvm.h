#pragma once
#include "pocl_cl.h"

#ifdef __cplusplus
extern "C" {
#endif

// compile kernel source to bitcode
int call_pocl_build( cl_device_id device, 
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

#ifdef __cplusplus

}
#endif

