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

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_ON((program->num_devices == 0),
    CL_INVALID_PROGRAM, "Invalid program (has no devices assigned)\n");

  POCL_RETURN_ERROR_ON((program->build_status == CL_BUILD_NONE),
    CL_INVALID_PROGRAM_EXECUTABLE, "You must call clBuildProgram first!"
      " (even for programs created with binaries)\n");

  POCL_RETURN_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
    CL_INVALID_PROGRAM_EXECUTABLE, "Last BuildProgram() was not successful\n");

  POCL_RETURN_ERROR_ON(((kernels != NULL && num_kernels == 0)
                       || (kernels == NULL && num_kernels != 0)),
                       CL_INVALID_VALUE, "kernels & num_kernels must be "
                       "either both set, or both NULL\n");

  POCL_RETURN_ERROR_ON((kernels && num_kernels < program->num_kernels),
                       CL_INVALID_VALUE,
                       "kernels is not NULL and num_kernels "
                       "is less than the number of kernels in program\n");

  for (idx = 0; idx < num_kernels; idx++)
    kernels[idx] = NULL;

  if (num_kernels > program->num_kernels)
    num_kernels = program->num_kernels;

  cl_int error_ret;
  if (num_kernels > 0 && kernels != NULL)
    {
      /* Create the kernels in the 'knames' list */
      for (idx = 0; idx < num_kernels; idx++)
        {
          kernels[idx] = POname(clCreateKernel) (program,
                                                 program->kernel_meta[idx].name,
                                                 &error_ret);

          /* Check for errors, clean up & bail.
           * If we happened to pass a invalid kernel name after all
           * that should be treated as a pocl bug, not user error.
           * TODO: what happens if the program is not valid?*/
          assert(error_ret != CL_INVALID_KERNEL_NAME);
          assert(error_ret != CL_INVALID_VALUE);
          if (error_ret != CL_SUCCESS)
            {
              for (; idx>0; idx--)
                POname(clReleaseKernel) (kernels[idx-1]);
              /* If error_ret is INVALID_KERNEL_DEFINITION, returning it here
               * is against the specification. But the specs doesn't say what to
               * do in such a case, and just returning it is the sanest thing
               * to do. */
              return error_ret;
            }
        }
    }

  if (num_kernels_ret)
    *num_kernels_ret = program->num_kernels;

  return CL_SUCCESS;
}
POsym(clCreateKernelsInProgram)
