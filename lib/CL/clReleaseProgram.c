/* OpenCL runtime library: clReleaseProgram()

   Copyright (c) 2011 Universidad Rey Juan Carlos, 
   Pekka Jääskeläinen / Tampere University of Technology
   
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

#include <string.h>
   
#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_runtime_config.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseProgram)(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  cl_kernel k;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RELEASE_OBJECT (program, new_refcount);

  if (new_refcount == 0)
    {

      /* Mark all kernels as having no program.
         FIXME: this should not be needed if the kernels
         retain the parent program (and release when the kernel
         is released). */
      for (k=program->kernels; k!=NULL; k=k->next)
        {          
          k->program = NULL;
        }

      if(program->devices != program->context->devices)
        POCL_MEM_FREE(program->devices);

      POCL_RELEASE_OBJECT (program->context, new_refcount);
      POCL_MEM_FREE(program->source);
      
      if (program->binaries != NULL)
        {
          POCL_MEM_FREE(program->binaries[0]);
          POCL_MEM_FREE(program->binaries);
        }
      POCL_MEM_FREE(program->binary_sizes);

      if ((!pocl_get_bool_option("POCL_KERNEL_CACHE", POCL_BUILD_KERNEL_CACHE)) &&
            (!pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0)) &&
            program->cache_dir)
        {
          pocl_remove_directory (program->cache_dir);
        }

      POCL_MEM_FREE(program->llvm_irs);
      POCL_MEM_FREE(program->cache_dir);
      POCL_MEM_FREE(program);
    }

  return CL_SUCCESS;
}
POsym(clReleaseProgram)
