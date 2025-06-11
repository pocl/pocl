/* OpenCL runtime library: clCreateProgramWithIL()

   Copyright (c) 2019 pocl developers
                 2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_program CL_API_CALL
POname(clCreateProgramWithIL)(cl_context context,
                              const void *il,
                              size_t length,
                              cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_2_1
{
  cl_program program = NULL;
  int errcode = CL_SUCCESS;
  char program_bc_spirv[POCL_MAX_PATHNAME_LENGTH];
  program_bc_spirv[0] = 0;

  POCL_GOTO_ERROR_COND ((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((il == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((length == 0), CL_INVALID_VALUE);

  int is_spirv = 0;
  int is_spirv_kernel
    = pocl_bitcode_is_spirv_execmodel_kernel ((const char *)il, length, 0);
  is_spirv += is_spirv_kernel;

  int is_spirv_shader
      = pocl_bitcode_is_spirv_execmodel_shader ((const char *)il, length);
  is_spirv += is_spirv_shader;

  POCL_GOTO_ERROR_ON (
      (!is_spirv), CL_INVALID_VALUE,
      "The IL provided to clCreateProgramWithIL "
      "is not recognized as SPIR-V!\n");

  /* TODO should we create a program for all devices ?
   * should we fail if we can't create for all devices ?
   * this seems to be unspecified.
   * right now, we create only for devices
   * which have supports_binary() callback
   * and fail if there's no such device in context. */
  unsigned num = context->num_devices;
  unsigned num_devices_with_spir = 0;
  cl_device_id *devices_with_spir
      = (cl_device_id *)alloca (num * sizeof (cl_device_id));

  for (unsigned i = 0; i < num; ++i)
    {
      cl_device_id dev = context->devices[i];
      if (dev->ops->supports_binary == NULL)
        continue;
      if (dev->ops->supports_binary (dev, length, il))
        {
          devices_with_spir[num_devices_with_spir++] = dev;
        }
    }

  POCL_GOTO_ERROR_ON ((num_devices_with_spir == 0), CL_INVALID_OPERATION,
                      "No device in context supports SPIR\n");

  program = create_program_skeleton (context, num_devices_with_spir,
                                     devices_with_spir, NULL, NULL, NULL,
                                     &errcode, CL_TRUE);
  if (program == NULL)
    goto ERROR;

  /* save the IL into cl_program, and find out spec constants */

  program->program_il = (char *)malloc (length);
  memcpy (program->program_il, il, length);
  program->program_il_size = length;
#ifdef ENABLE_SPIRV
  /* this might change the size */
  pocl_preprocess_spirv_input (program);
#endif

  errcode = pocl_cache_write_spirv (program_bc_spirv,
                                    program->program_il,
                                    program->program_il_size);

#ifdef ENABLE_SPIRV
  pocl_get_program_spec_constants (program, program_bc_spirv,
                                   program->program_il,
                                   program->program_il_size);

#endif


ERROR:
  if (errcode_ret)
    *errcode_ret = errcode;
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0)
    {
      if (program_bc_spirv[0] != 0)
        pocl_remove (program_bc_spirv);
    }
  return program;
}
POsym(clCreateProgramWithIL)
