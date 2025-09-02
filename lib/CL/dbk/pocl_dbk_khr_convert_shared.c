/* pocl_dkb_khr_convert_shared.h - generic element type convert
   defined builtin kernel functions.

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
#include "pocl_dbk_khr_convert_shared.h"

#include "pocl_tensor_util.h"

int pocl_validate_convert_attrs(cl_dbk_id_exp kernel_id,
                                const void *kernel_attributes) {
  assert(kernel_id == CL_DBK_CONVERT_EXP);
  const cl_dbk_attributes_convert_exp *cvt_attrs = kernel_attributes;

  POCL_RETURN_ERROR_ON(
      cvt_attrs->src.rank != cvt_attrs->dst.rank, CL_INVALID_TENSOR_RANK_EXP,
      "convert_exp tensor operands must have the same rank.\n");

  POCL_RETURN_ERROR_ON(
      !pocl_tensor_shape_equals(&cvt_attrs->src, &cvt_attrs->dst),
      CL_INVALID_TENSOR_SHAPE_EXP,
      "convert_exp tensor operands must have the same shape.\n");

  POCL_RETURN_ERROR_ON(
      cvt_attrs->src.dtype == cvt_attrs->dst.dtype,
      CL_INVALID_TENSOR_DATATYPE_EXP,
      "convert_exp tensor operands must have different data type.");

  return CL_SUCCESS;
}

int pocl_release_convert_attrs(cl_dbk_id_exp kernel_id,
                               void *kernel_attributes) {
  assert(kernel_id == CL_DBK_CONVERT_EXP);
  cl_dbk_attributes_convert_exp *cvt_attrs = kernel_attributes;

  pocl_tensor_destroy_body (&cvt_attrs->src);
  pocl_tensor_destroy_body (&cvt_attrs->dst);
  POCL_MEM_FREE (cvt_attrs);
  return CL_SUCCESS;
}

void *pocl_copy_convert_attrs(cl_dbk_id_exp kernel_id,
                              const void *kernel_attributes) {
  assert(kernel_id == CL_DBK_CONVERT_EXP);
  const cl_dbk_attributes_convert_exp *cvt_attrs = kernel_attributes;

  cl_dbk_attributes_convert_exp *copy =
      calloc(1, sizeof(cl_dbk_attributes_convert_exp));
  if (!copy)
    return NULL;

  if (pocl_tensor_copy(&copy->src, &cvt_attrs->src))
    goto ERROR;

  if (pocl_tensor_copy(&copy->dst, &cvt_attrs->dst))
    goto ERROR;

  return copy;

 ERROR:
  pocl_release_convert_attrs(kernel_id, copy);
  return NULL;
}
