/* pocl_dkb_set_rows_shared.h - DBK modeling GGML's set_rows operation.

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
#include "pocl_dbk_set_rows_shared.h"

#include "pocl_tensor_util.h"

#define REQUIRE_ATTR_EQUAL(_lhs, _rhs, _errcode)                              \
  do                                                                          \
    {                                                                         \
      POCL_RETURN_ERROR_ON ((attrs->_lhs) != (_rhs), (_errcode),              \
                            "set_rows_exp: must '" #_lhs "' == " #_rhs ".");  \
    }                                                                         \
  while (0)

#define REQUIRE_ATTRS_EQUAL(_lhs, _rhs, _errcode)                             \
  do                                                                          \
    {                                                                         \
      POCL_RETURN_ERROR_ON ((attrs->_lhs) != (attrs->_rhs), (_errcode),       \
                            "set_rows_exp: must '" #_lhs "' == '" #_rhs       \
                            "'.");                                            \
    }                                                                         \
  while (0)

int
pocl_validate_set_rows_attrs (cl_dbk_id_exp kernel_id,
                              const void *kernel_attributes)
{
  assert (kernel_id == CL_DBK_SET_ROWS_EXP);
  const cl_dbk_attributes_set_rows_exp *attrs = kernel_attributes;

  REQUIRE_ATTR_EQUAL (data_in.rank, 4, CL_INVALID_TENSOR_SHAPE_EXP);
  REQUIRE_ATTR_EQUAL (rows.rank, 4, CL_INVALID_TENSOR_SHAPE_EXP);
  REQUIRE_ATTR_EQUAL (indices.rank, 4, CL_INVALID_TENSOR_SHAPE_EXP);
  REQUIRE_ATTR_EQUAL (data_out.rank, 4, CL_INVALID_TENSOR_SHAPE_EXP);

  POCL_RETURN_ERROR_ON (
    !pocl_tensor_shape_equals (&attrs->data_in, &attrs->data_out),
    CL_INVALID_TENSOR_SHAPE_EXP,
    "set_rows_exp: shapes of data_in and data_out must match.");

  REQUIRE_ATTRS_EQUAL (rows.shape[0], data_in.shape[0],
                       CL_INVALID_TENSOR_SHAPE_EXP);
  REQUIRE_ATTRS_EQUAL (rows.shape[1], data_in.shape[1],
                       CL_INVALID_TENSOR_SHAPE_EXP);
  REQUIRE_ATTRS_EQUAL (rows.shape[3], data_in.shape[3],
                       CL_INVALID_TENSOR_SHAPE_EXP);
  REQUIRE_ATTRS_EQUAL (indices.shape[3], rows.shape[2],
                       CL_INVALID_TENSOR_SHAPE_EXP);

  POCL_RETURN_ERROR_ON (attrs->data_in.shape[0] % attrs->indices.shape[1] != 0,
                        CL_INVALID_TENSOR_SHAPE_EXP,
                        "set_rows_exp: data_in.shape[0] must be divisable "
                        "by indicies.shape[1]");
  POCL_RETURN_ERROR_ON (attrs->data_in.shape[1] % attrs->indices.shape[2] != 0,
                        CL_INVALID_TENSOR_SHAPE_EXP,
                        "set_rows_exp: data_in.shape[1] must be divisable "
                        "by indicies.shape[2]");

  POCL_RETURN_ERROR_ON (attrs->rows.dtype != CL_TENSOR_DTYPE_FP32_EXP,
                        CL_INVALID_TENSOR_DATATYPE_EXP,
                        "set_rows_exp: element type of 'rows' must be "
                        "CL_TENSOR_DTYPE_FP32_EXP.");
  POCL_RETURN_ERROR_ON (attrs->indices.dtype != CL_TENSOR_DTYPE_INT64_EXP,
                        CL_INVALID_TENSOR_DATATYPE_EXP,
                        "set_rows_exp: element type of 'indices' must be "
                        "CL_TENSOR_DTYPE_INT64_EXP.");

  return CL_SUCCESS;
}

int
pocl_release_set_rows_attrs (cl_dbk_id_exp kernel_id, void *kernel_attributes)
{
  assert (kernel_id == CL_DBK_SET_ROWS_EXP);
  cl_dbk_attributes_set_rows_exp *set_rows_attrs = kernel_attributes;

  pocl_tensor_destroy_body (&set_rows_attrs->data_in);
  pocl_tensor_destroy_body (&set_rows_attrs->rows);
  pocl_tensor_destroy_body (&set_rows_attrs->indices);
  pocl_tensor_destroy_body (&set_rows_attrs->data_out);
  POCL_MEM_FREE (set_rows_attrs);
  return CL_SUCCESS;
}

void *
pocl_copy_set_rows_attrs (cl_dbk_id_exp kernel_id,
                          const void *kernel_attributes)
{
  assert (kernel_id == CL_DBK_SET_ROWS_EXP);
  const cl_dbk_attributes_set_rows_exp *set_rows_attrs = kernel_attributes;

  cl_dbk_attributes_set_rows_exp *copy
    = calloc (1, sizeof (cl_dbk_attributes_set_rows_exp));
  if (!copy)
    return NULL;

  if (pocl_tensor_copy (&copy->data_in, &set_rows_attrs->data_in))
    goto ERROR;

  if (pocl_tensor_copy (&copy->rows, &set_rows_attrs->rows))
    goto ERROR;

  if (pocl_tensor_copy (&copy->indices, &set_rows_attrs->indices))
    goto ERROR;

  if (pocl_tensor_copy (&copy->data_out, &set_rows_attrs->data_out))
    goto ERROR;

  return copy;

ERROR:
  pocl_release_set_rows_attrs (kernel_id, copy);
  return NULL;
}
