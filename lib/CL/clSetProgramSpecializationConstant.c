/* OpenCL runtime library: clSetProgramSpecializationConstant()

   Copyright (c) 2022 Michal Babej / Tampere University

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

#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetProgramSpecializationConstant)
                                  (cl_program  program,
                                   cl_uint     spec_id,
                                   size_t      spec_size,
                                   const void* spec_value) CL_API_SUFFIX__VERSION_2_2
{
  /* if SPIR-V is disabled, return early */
#if defined(ENABLE_CONFORMANCE) && !defined(ENABLE_SPIRV)
  return CL_INVALID_OPERATION;
#endif

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  assert (program->num_devices != 0);

  POCL_RETURN_ERROR_ON (
      (program->program_il == NULL || program->program_il_size == 0),
      CL_INVALID_PROGRAM, "The program does not contain IL\n");

  for (unsigned i = 0; i < program->num_spec_consts; ++i)
    {
      if (program->spec_const_ids[i] == spec_id)
        {
          POCL_RETURN_ERROR_ON ((program->spec_const_sizes[i] != spec_size),
                                CL_INVALID_VALUE,
                                "Given spec constant size (%zu)"
                                "doesn't match the expected (%u)\n",
                                spec_size, program->spec_const_sizes[i]);
          program->spec_const_values[i] = 0;
          memcpy ((char *)&program->spec_const_values[i], spec_value,
                  spec_size);
          program->spec_const_is_set[i] = CL_TRUE;
          return CL_SUCCESS;
        }
    }

  POCL_RETURN_ERROR (CL_INVALID_SPEC_ID,
                     "Unknown specialization constant ID %u\n", spec_id);
}
POsym (clSetProgramSpecializationConstant)
