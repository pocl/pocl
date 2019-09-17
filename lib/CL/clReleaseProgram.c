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
#include "pocl_cache.h"
#include "pocl_llvm.h"
#include "devices.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseProgram)(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  unsigned i, j;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RELEASE_OBJECT (program, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release program %p, new refcount: %d, kernel #: %zu \n", program, new_refcount, program->num_kernels);

  if (new_refcount == 0)
    {
      cl_context context = program->context;
      POCL_MSG_PRINT_REFCOUNTS ("Free program %p\n", program);

      /* there should be no kernels left when we're releasing the program */
      assert (program->kernels == NULL);

      if (program->devices != program->context->devices)
        POCL_MEM_FREE(program->devices);

      POCL_MEM_FREE(program->source);

      POCL_MEM_FREE (program->program_il);

      POCL_MEM_FREE(program->binary_sizes);
      if (program->binaries)
        for (i = 0; i < program->num_devices; ++i)
          POCL_MEM_FREE(program->binaries[i]);
      POCL_MEM_FREE(program->binaries);

      POCL_MEM_FREE(program->pocl_binary_sizes);
      if (program->pocl_binaries)
        for (i = 0; i < program->num_devices; ++i)
          POCL_MEM_FREE(program->pocl_binaries[i]);
      POCL_MEM_FREE(program->pocl_binaries);

      pocl_cache_cleanup_cachedir(program);

      if (program->build_log)
        for (i = 0; i < program->num_devices; ++i)
          POCL_MEM_FREE(program->build_log[i]);
      POCL_MEM_FREE(program->build_log);

      if (program->num_kernels)
        {
          for (i = 0; i < program->num_kernels; i++)
            {
              pocl_kernel_metadata_t *meta = &program->kernel_meta[i];
              POCL_MEM_FREE (meta->attributes);
              POCL_MEM_FREE (meta->name);
              for (j = 0; j < meta->num_args; ++j)
                {
                  POCL_MEM_FREE (meta->arg_info[j].name);
                  POCL_MEM_FREE (meta->arg_info[j].type_name);
                }
              POCL_MEM_FREE (meta->arg_info);
              if (meta->data != NULL)
                for (j = 0; j < program->num_devices; ++j)
                  if (meta->data[j] != NULL)
                    meta->data[j] = NULL; // TODO free data in driver callback
              POCL_MEM_FREE (meta->data);
              POCL_MEM_FREE (meta->local_sizes);
              POCL_MEM_FREE (meta->build_hash);
            }
          POCL_MEM_FREE (program->kernel_meta);
        }

      POCL_MEM_FREE(program->build_hash);
      POCL_MEM_FREE(program->compiler_options);

#ifdef ENABLE_LLVM
      if (program->llvm_irs)
        for (i = 0; i < program->num_devices; ++i)
          pocl_free_llvm_irs (program, i);
#endif

      POCL_MEM_FREE(program->llvm_irs);
      POCL_DESTROY_OBJECT (program);
      POCL_MEM_FREE(program);

      POname(clReleaseContext)(context);
    }

  return CL_SUCCESS;
}
POsym(clReleaseProgram)
