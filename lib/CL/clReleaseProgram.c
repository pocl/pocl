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

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_cache.h"
#include "pocl_llvm.h"
#include "devices.h"

extern unsigned long program_c;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseProgram)(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  unsigned i, j;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  POCL_RELEASE_OBJECT (program, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS (
      "Release Program %" PRId64 " (%p), Refcount: %d, Kernel #: %zu \n",
      program->id, program, new_refcount, program->num_kernels);

  if (new_refcount == 0)
    {
      VG_REFC_ZERO (program);

      POCL_ATOMIC_DEC (program_c);

      cl_context context = program->context;
      POCL_MSG_PRINT_REFCOUNTS ("Free Program %" PRId64 " (%p)\n", program->id,
                                program);
      TP_FREE_PROGRAM (context->id, program->id);

      /* there should be no kernels left when we're releasing the program */
      assert (program->kernels == NULL);

      for (i = 0; i < program->num_devices; ++i)
        {
          cl_device_id device = program->devices[i];
          if (device->ops->free_program)
            device->ops->free_program (device, program, i);
        }

      if (program->devices != program->context->devices
          && program->devices != program->associated_devices)
        POCL_MEM_FREE(program->devices);
      if (program->associated_devices != program->context->devices)
        POCL_MEM_FREE (program->associated_devices);

      POCL_MEM_FREE(program->source);

      POCL_MEM_FREE (program->program_il);
      POCL_MEM_FREE (program->spec_const_ids);
      POCL_MEM_FREE (program->spec_const_is_set);
      POCL_MEM_FREE (program->spec_const_sizes);
      POCL_MEM_FREE (program->spec_const_values);

      POCL_MEM_FREE(program->binary_sizes);
      if (program->binaries)
        for (i = 0; i < program->associated_num_devices; ++i)
          POCL_MEM_FREE(program->binaries[i]);
      POCL_MEM_FREE(program->binaries);

      POCL_MEM_FREE(program->pocl_binary_sizes);
      if (program->pocl_binaries)
        for (i = 0; i < program->associated_num_devices; ++i)
          POCL_MEM_FREE(program->pocl_binaries[i]);
      POCL_MEM_FREE(program->pocl_binaries);

      pocl_cache_cleanup_cachedir(program);

      if (program->build_log)
        for (i = 0; i < program->associated_num_devices; ++i)
          POCL_MEM_FREE(program->build_log[i]);
      POCL_MEM_FREE(program->build_log);

      for (i = 0; i < program->num_kernels; i++)
        pocl_free_kernel_metadata (program, i);
      POCL_MEM_FREE (program->kernel_meta);

      POCL_MEM_FREE (program->build_hash);
      POCL_MEM_FREE (program->compiler_options);
      POCL_MEM_FREE (program->data);
      POCL_MEM_FREE (program->global_var_total_size);
      POCL_MEM_FREE (program->llvm_irs);
      POCL_MEM_FREE (program->gvar_storage);

      for (i = 0; i < program->num_builtin_kernels; ++i)
        POCL_MEM_FREE (program->builtin_kernel_names[i]);
      POCL_MEM_FREE (program->builtin_kernel_names);
      POCL_MEM_FREE (program->concated_builtin_names);

      POCL_DESTROY_OBJECT (program);
      POCL_MEM_FREE (program);

      POname(clReleaseContext)(context);
    }
  else
    {
      VG_REFC_NONZERO (program);
    }

  return CL_SUCCESS;
}
POsym(clReleaseProgram)
