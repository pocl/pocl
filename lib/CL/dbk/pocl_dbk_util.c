/* pocl_dbk_util.c - utility functions for DBK descriptions.

   Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy

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

#include "pocl_dbk_util.h"

POCL_EXPORT
cl_int
pocl_dbk_unpack_bin_operands (cl_dbk_id_exp kernel_id,
                              const void *kernel_attributes,
                              const cl_tensor_desc_exp *ops_out[3])
{
  assert (kernel_attributes);
  assert (ops_out);

#define HANDLE_CASE(_ID, _STRUCT)                                             \
  case _ID:                                                                   \
    do                                                                        \
      {                                                                       \
        _STRUCT *bin_dbk = (_STRUCT *)kernel_attributes;                      \
        ops_out[0] = &bin_dbk->src0;                                          \
        ops_out[1] = &bin_dbk->src1;                                          \
        ops_out[2] = &bin_dbk->dst;                                           \
        return CL_SUCCESS;                                                    \
      }                                                                       \
    while (0)

  switch (kernel_id)
    {
    default:
      assert (!"UNREACHABLE!");
      return CL_DBK_INVALID_ID_EXP;

      HANDLE_CASE (CL_DBK_ADD_EXP, cl_dbk_attributes_add_exp);
      HANDLE_CASE (CL_DBK_MUL_EXP, cl_dbk_attributes_mul_exp);
    }
#undef HANDLE_CASE
}
