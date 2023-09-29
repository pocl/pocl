/* OpenCL runtime library: clCreateProgramWithIL()

   Copyright (c) 2019 pocl developers

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
#include "pocl_util.h"
#include "pocl_shared.h"

/* max number of lines in output of 'llvm-spirv --spec-const-info' */
#define MAX_SPEC_CONSTANT_LINES 4096
/* max bytes in output of 'llvm-spirv --spec-const-info' */
#define MAX_OUTPUT_BYTES 65536

#ifdef ENABLE_SPIRV
static int
get_program_spec_constants (cl_program program, char *program_bc_spirv)
{
  char *args[] = { LLVM_SPIRV, "--spec-const-info", program_bc_spirv, NULL };
  char captured_output[MAX_OUTPUT_BYTES];
  size_t captured_bytes = MAX_OUTPUT_BYTES;
  int errcode = CL_SUCCESS;

  errcode = pocl_run_command_capture_output (captured_output, &captured_bytes,
                                             args);
  POCL_RETURN_ERROR_ON ((errcode != 0), CL_INVALID_BINARY, "External command "
                        "(llvm-spirv --spec-const-info) failed!\n");

  captured_output[captured_bytes] = 0;
  char *lines[MAX_SPEC_CONSTANT_LINES];
  unsigned num_lines = 0;
  char delim[2] = { 0x0A, 0x0 };
  char *token = strtok (captured_output, delim);
  while (num_lines < MAX_SPEC_CONSTANT_LINES && token != NULL)
    {
      lines[num_lines++] = strdup (token);
      token = strtok (NULL, delim);
    }
  POCL_GOTO_ERROR_ON ((num_lines == 0 || num_lines >= MAX_SPEC_CONSTANT_LINES),
                      CL_INVALID_BINARY,
                      "Can't parse output from llvm-spirv\n");

  unsigned num_const = 0;
  int r = sscanf (
      lines[0], "Number of scalar specialization constants in the module = %u",
      &num_const);
  POCL_GOTO_ERROR_ON ((r < 1 || num_const > num_lines), CL_INVALID_BINARY,
                      "Can't parse first line of output");

  program->num_spec_consts = num_const;
  if (num_const > 0)
    {
      program->spec_const_ids = calloc (num_const, sizeof (cl_uint));
      program->spec_const_sizes = calloc (num_const, sizeof (cl_uint));
      program->spec_const_values = calloc (num_const, sizeof (uint64_t));
      program->spec_const_is_set = calloc (num_const, sizeof (char));
      for (unsigned i = 0; i < program->num_spec_consts; ++i)
        {
          unsigned spec_id, spec_size;
          int r
              = sscanf (lines[i + 1], "Spec const id = %u, size in bytes = %u",
                        &spec_id, &spec_size);
          POCL_GOTO_ERROR_ON ((r < 2), CL_INVALID_BINARY,
                              "Can't parse %u-th line of output:\n%s\n",
                              i+1, lines[i+1]);
          program->spec_const_ids[i] = spec_id;
          program->spec_const_sizes[i] = spec_size;
          program->spec_const_values[i] = 0;
          program->spec_const_is_set[i] = CL_FALSE;
        }
    }
  errcode = CL_SUCCESS;
ERROR:
  for (unsigned i = 0; i < num_lines; ++i)
    free (lines[i]);
  if (errcode != CL_SUCCESS)
    {
      program->num_spec_consts = 0;
    }
  return errcode;
}
#endif


CL_API_ENTRY cl_program CL_API_CALL
POname(clCreateProgramWithIL)(cl_context context,
                              const void *il,
                              size_t length,
                              cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_2_1
{
  /* if SPIR-V is disabled, return error early */
#if defined(ENABLE_CONFORMANCE) && !defined(ENABLE_SPIRV)
  if (errcode_ret)
    *errcode_ret = CL_INVALID_OPERATION;
  return NULL;
#endif
  cl_program program = NULL;
  int errcode = CL_SUCCESS;
  char program_bc_spirv[POCL_MAX_PATHNAME_LENGTH];
  program_bc_spirv[0] = 0;

  POCL_GOTO_ERROR_COND ((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((il == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((length == 0), CL_INVALID_VALUE);

  int is_spirv = 0;
#ifdef ENABLE_SPIRV
  int is_spirv_kernel
      = pocl_bitcode_is_spirv_execmodel_kernel ((const char *)il, length);
  is_spirv += is_spirv_kernel;
#endif
#ifdef ENABLE_VULKAN
  int is_spirv_shader
      = pocl_bitcode_is_spirv_execmodel_shader ((const char *)il, length);
  is_spirv += is_spirv_shader;
#endif

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
          devices_with_spir[num_devices_with_spir] = dev;
          ++num_devices_with_spir;
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
  errcode = pocl_cache_write_spirv (program_bc_spirv, (const char *)il,
                                    (uint64_t)length);
#ifdef ENABLE_SPIRV
  get_program_spec_constants (program, program_bc_spirv);
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
