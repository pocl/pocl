#include "config.h"
#include "install-paths.h"
#include "pocl_llvm.h"
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include "pocl_cl.h"


// TODO: copies...
#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 128

int call_pocl_build( cl_device_id device, 
                     const char* source_file_name,
                     const char* binary_file_name,
                     const char* device_tmpdir,
                     const char* user_options )
{
  int error;
  char *pocl_build_script;
  char command[COMMAND_LENGTH];

  if (getenv("POCL_BUILDING") != NULL)
    pocl_build_script = BUILDDIR "/scripts/" POCL_BUILD;
  else if (access(PKGDATADIR "/" POCL_BUILD, X_OK) == 0)
    pocl_build_script = PKGDATADIR "/" POCL_BUILD;
  else
    pocl_build_script = POCL_BUILD;

  if (device->llvm_target_triplet != NULL)
    {
      error = snprintf(command, COMMAND_LENGTH,
                       "USER_OPTIONS=\"%s\" %s -t %s -o %s %s", 
                       user_options,
                       pocl_build_script,
                       device->llvm_target_triplet,                               
                       binary_file_name, source_file_name);
     }
  else 
    {
      error = snprintf(command, COMMAND_LENGTH,
                       "USER_OPTIONS=\"%s\" %s -o %s %s", 
                      user_options,
                      pocl_build_script,
                      binary_file_name, source_file_name);
    }
 
  if (error < 0)
    return CL_OUT_OF_HOST_MEMORY;

  /* call the customized build command, if needed for the
     device driver */
  if (device->build_program != NULL)
    {
      error = device->build_program 
        (device->data, source_file_name, binary_file_name, 
         command, device_tmpdir);
    }
  else
    {
       error = system(command);
    }

  return error;
}

