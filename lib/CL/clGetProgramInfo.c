/* OpenCL runtime library: clGetProgramInfo()

   Copyright (c) 2011 Erik Schnetter
   
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
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "pocl_cache.h"
#include "pocl_binary.h"
#include "pocl_shared.h"

cl_int
program_compile_dynamic_wg_binaries(cl_program program);

/******************************************************************************/

static void get_binary_sizes(cl_program program, size_t *sizes)
{
#ifdef OCS_AVAILABLE
  if (program_compile_dynamic_wg_binaries(program) != CL_SUCCESS)
    {
      memset(sizes, 0, program->num_devices * sizeof(size_t));
      return;
    }
#endif
  unsigned i;
  for (i=0; i < program->num_devices; i++)
    {
      if (!program->pocl_binaries[i] && program->binaries[i])
        program->pocl_binary_sizes[i] = pocl_binary_sizeof_binary(program, i);
      if (program->pocl_binaries[i])
        sizes[i] = program->pocl_binary_sizes[i];
      else
        sizes[i] = 0;
    }
}

static void get_binaries(cl_program program, unsigned char **binaries)
{
#ifdef OCS_AVAILABLE
  if (program_compile_dynamic_wg_binaries(program) != CL_SUCCESS)
    {
      memset(binaries, 0, program->num_devices * sizeof(unsigned char*));
      return;
    }
#endif
  unsigned i;
  size_t res;
  for (i=0; i < program->num_devices; i++)
    {
      if (!program->pocl_binaries[i] && program->binaries[i])
        {
          pocl_binary_serialize(program, i, &res);
          if (program->pocl_binary_sizes[i])
            assert(program->pocl_binary_sizes[i] == res);
          program->pocl_binary_sizes[i] = res;
        }
      if (program->pocl_binaries[i])
        memcpy(binaries[i], program->pocl_binaries[i], program->pocl_binary_sizes[i]);
      else
        binaries[i] = NULL;
    }
}

/******************************************************************************/

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetProgramInfo)(cl_program program,
                 cl_program_info param_name,
                 size_t param_value_size,
                 void *param_value,
                 size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i;
  
  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  switch (param_name)
  {
  case CL_PROGRAM_REFERENCE_COUNT:
    POCL_RETURN_GETINFO(cl_uint, (cl_uint)program->pocl_refcount);
  case CL_PROGRAM_CONTEXT:
    POCL_RETURN_GETINFO(cl_context, program->context);

  case CL_PROGRAM_SOURCE:
    {
      const char *source = program->source;
      if (source == NULL)
        source = "";

      POCL_RETURN_GETINFO_STR(source);
    }

  case CL_PROGRAM_BINARY_SIZES:
    {
      POCL_RETURN_ERROR_COND(program->build_status != CL_BUILD_SUCCESS,
                             CL_INVALID_PROGRAM);
      size_t const value_size = sizeof(size_t) * program->num_devices;
      POCL_RETURN_GETINFO_INNER(value_size, get_binary_sizes(program, param_value));
    }

  case CL_PROGRAM_BINARIES:
    {
      POCL_RETURN_ERROR_COND(program->build_status != CL_BUILD_SUCCESS,
                             CL_INVALID_PROGRAM);
      size_t const value_size = sizeof(unsigned char *) * program->num_devices;
      POCL_RETURN_GETINFO_INNER(value_size, get_binaries(program, param_value));
    }

  case CL_PROGRAM_NUM_DEVICES:
    POCL_RETURN_GETINFO(cl_uint, program->num_devices);

  case CL_PROGRAM_DEVICES:
    {
      size_t const value_size = sizeof(cl_device_id) * program->num_devices;
      POCL_RETURN_GETINFO_SIZE(value_size, program->devices);
    }

  case CL_PROGRAM_NUM_KERNELS:
    {
      POCL_RETURN_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
                           CL_INVALID_PROGRAM_EXECUTABLE,
                           "This information is only available after a "
                           "successful program executable has been built\n");
      POCL_RETURN_GETINFO(size_t, program->num_kernels);
    }

  case CL_PROGRAM_KERNEL_NAMES:
    {
      POCL_RETURN_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
                           CL_INVALID_PROGRAM_EXECUTABLE,
                           "This information is only available after a "
                           "successful program executable has been built\n");

       /* Note: In the specification (2.0) of the other XXXInfo
         functions, param_value_size_ret is described as follows:

         > param_value_size_ret returns the actual size in bytes of data
         > being *queried* by param_value.

         while in GetProgramInfo and APIs defined later in the
         documentation, it is:

         > param_value_size_ret returns the actual size in bytes of data
         > *copied* to param_value.

         it reads as if the spec allows the implementation to stop copying
         the string at an arbitrary point where the limit
         (param_value_size) is reached, but that's not the case. When it
         happens, it should instead raise an error CL_INVALID_VALUE.

         Also note the specification of the param_value_size_ret to param_name
         CL_PROGRAM_SOURCE.  It says "The actual number of characters that
         represents[sic] the program source code including the null terminator
         is returned in param_value_size_ret." By an analogy, it is sane to
         return the size of entire concatenated string, not the size of
         bytes copied (partially).

         Also note the specification of GetPlatformInfo +
         CL_PLATFORM_EXTENSIONS.  it refers to "param_value_size_ret" as
         the actual size in bytes of data being *queried*, and its
         description of param_value_size is the same.

         --- guicho271828
      */
      size_t num_kernels = program->num_kernels;
      size_t size = 0;

      /* optimized for clarity */
      for (i = 0; i < num_kernels; ++i)
        {
          size += strlen (program->kernel_meta[i].name);
          if (i != num_kernels - 1)
            size += 1;          /* a semicolon */
        }
      size += 1;                /* a NULL */
      if (param_value_size_ret)
        *param_value_size_ret = size;
      if (param_value)
        {
          /* only when param_value is non-NULL */
          if (size > param_value_size)
            return CL_INVALID_VALUE;
          /* should not break from the switch clause
             because of POCL_ABORT_UNIMPLEMENTED */
          for (i = 0; i < num_kernels; ++i)
            {
              if (i == 0)
                strcpy (param_value, program->kernel_meta[i].name); /* copy including NULL */
              else
                strcat ((char*)param_value, program->kernel_meta[i].name);
              if (i != num_kernels - 1)
                strcat ((char*)param_value, ";");
            }
        }
      return CL_SUCCESS;
    }
  default:
    POCL_RETURN_ERROR_ON(1, CL_INVALID_VALUE,
                         "Parameter %i not implemented\n", param_name);
  }

}
POsym(clGetProgramInfo)
