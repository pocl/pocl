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
  /* SPIR is disabled when building with conformance */
#ifdef ENABLE_CONFORMANCE
  return CL_INVALID_OPERATION;
#endif

#ifdef ENABLE_VULKAN
  /* TODO implement for Vulkan + possibly CPU devices */
#endif

  int errcode = CL_SUCCESS;
  size_t i;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  assert (program->num_devices != 0);

  POCL_RETURN_ERROR_ON (
      (program->program_il == NULL || program->program_il_size == 0),
      CL_INVALID_PROGRAM, "The program does not contain IL\n");

  return CL_INVALID_OPERATION;
}
POsym (clSetProgramSpecializationConstant)
