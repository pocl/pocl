/* OpenCL runtime library: clCreateKernelsInProgram 
 * 
 * Author: Kalle Raiskila 2014.
 * This file is in the public domain. 
 */

#include "pocl_cl.h"
#include "pocl_llvm.h"
#include "pocl_intfn.h"


CL_API_ENTRY cl_int CL_API_CALL
POname(clCreateKernelsInProgram)(cl_program      program ,
                         cl_uint         num_kernels ,
                         cl_kernel *     kernels ,
                         cl_uint *       num_kernels_ret ) CL_API_SUFFIX__VERSION_1_0
{
  unsigned idx;
  unsigned num_kern_found;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_ON((program->num_devices == 0),
    CL_INVALID_PROGRAM, "Invalid program (has no devices assigned)\n");

  POCL_RETURN_ERROR_ON((program->build_status == CL_BUILD_NONE),
    CL_INVALID_PROGRAM_EXECUTABLE, "You must call clBuildProgram first!"
      " (even for programs created with binaries)\n");

  POCL_RETURN_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
    CL_INVALID_PROGRAM_EXECUTABLE, "Last BuildProgram() was not successful\n")

  POCL_RETURN_ERROR_ON((program->llvm_irs == NULL),
    CL_INVALID_PROGRAM_EXECUTABLE, "No built binaries in program "
    "(this shouldn't happen...)\n");

  num_kern_found = pocl_llvm_get_kernel_count(program);

  POCL_RETURN_ERROR_ON((kernels && num_kernels < num_kern_found), CL_INVALID_VALUE,
      "kernels is not NULL and num_kernels "
      "is less than the number of kernels in program");

  if (num_kern_found > 0 && kernels != NULL)
    {
      /* Get list of kernel names in program */
      const char** knames = (const char**)calloc( num_kernels, sizeof(const char *) );
      if (knames == NULL)
        return CL_OUT_OF_HOST_MEMORY;
      pocl_llvm_get_kernel_names (program, knames, num_kernels);

      /* Create the kernels in the 'knames' list */
      for (idx = 0; idx < num_kern_found; idx++)
        {
          cl_int error_ret;
          kernels[idx] = clCreateKernel (program, knames[idx], &error_ret);

          /* Check for errors, clean up & bail.
           * If we happened to pass a invalid kernel name after all
           * that should be treated as a pocl bug, not user error.
           * TODO: what happens if the program is not valid?*/
          assert(error_ret != CL_INVALID_KERNEL_NAME);
          assert(error_ret != CL_INVALID_VALUE);
          if (error_ret != CL_SUCCESS)
            {
              for (; idx>0; idx--)
                {
                  clReleaseKernel (kernels[idx-1]);
                }
              POCL_MEM_FREE(knames);
              /* If error_ret is INVALID_KERNEL_DEFINITION, returning it here
               * is against the specification. But the specs doesn't say what to
               * do in such a case, and just returning it is the sanest thing
               * to do. */
              return error_ret;
            }
        }
      for (idx = num_kern_found ; idx < num_kernels; idx++)
        {
          kernels[idx] = NULL;
        }
      POCL_MEM_FREE(knames);
    }

  if (num_kernels_ret)
    *num_kernels_ret = num_kern_found;

  return CL_SUCCESS;
}
POsym(clCreateKernelsInProgram)
