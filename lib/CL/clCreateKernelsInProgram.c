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
  int idx;
  int num_kern_found;
  
  /* Get list of kernel names in program */
  const char** knames = (const char**)malloc( num_kernels*sizeof(const char *) ); 
  if (knames == NULL) 
    return CL_OUT_OF_HOST_MEMORY;
  num_kern_found = pocl_llvm_get_kernel_names (program, knames, num_kernels);

  /* Sanity & quick return checks that the specs define */
  if (num_kernels_ret != NULL)
    *num_kernels_ret = num_kern_found;
  if (kernels ==  NULL) 
    {
      POCL_MEM_FREE(knames);
      return CL_SUCCESS;
    }

  POCL_RETURN_ERROR_ON((num_kernels < num_kern_found), CL_INVALID_VALUE,
      "kernels is not NULL and num_kernels "
      "is less than the number of kernels in program");

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
              POCL_MEM_FREE(knames);
              /* If error_ret is INVALID_KERNEL_DEFINITION, returning it here
               * is against the specification. But the specs doesn't say what to 
               * do in such a case, and just returning it is the sanest thing 
               * to do. */
              return error_ret; 
            }
        }
    }
  
  POCL_MEM_FREE(knames);
  return CL_SUCCESS;
}
POsym(clCreateKernelsInProgram)
