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


#ifdef __cplusplus
}
#endif

