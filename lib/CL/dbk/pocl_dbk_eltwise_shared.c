/* pocl_dbk_eltwise_shared.c - element-wise DBKs

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

#include "pocl_dbk_eltwise_shared.h"

#include "pocl_dbk_util.h"
#include "pocl_tensor_util.h"

/** Return a tensor shape that is result of numpy-flavored implicit
 * broadcasting of the given two tensors.
 *
 * Broadcasting rules:
 * https://numpy.org/devdocs/user/basics.broadcasting.html#general-broadcasting-rules.
 *
 * E.g. t0->shape = {1, 4}, t1->shape{2, 3, 1} --> {2, 3, 4}.
 *
 * The array length of 'shape_out' must be max(t0->rank, t1->rank).
 *
 * On success, return resulting shape via 'shape_out' reference and
 * return rank of it. On a failure, zero is returned and shape_out array
 * has undefined values.
 */
static unsigned
get_shape_for_implicit_broadcast (const cl_tensor_desc_exp *t0,
                                  const cl_tensor_desc_exp *t1,
                                  cl_tensor_shape_exp *shape_out)
{
  assert (t0);
  assert (t1);
  assert (t0->rank);
  assert (t1->rank);
  assert (shape_out);

  unsigned result_rank = max (t0->rank, t1->rank);
  for (int i = -1; i >= -(int)result_rank; i--)
    {
      size_t t0_dim = pocl_tensor_dim_size_or (t0, i, 1);
      size_t t1_dim = pocl_tensor_dim_size_or (t1, i, 1);

      if (!(t0_dim == t1_dim || t0_dim == 1 || t1_dim == 1))
        return 0;

      shape_out[result_rank + i] = max (t0_dim, t1_dim);
    }

  return result_rank;
}

int
pocl_validate_eltwise_binary_dbk_attrs (cl_dbk_id_exp kernel_id,
                                        const void *kernel_attributes)
{

  const cl_tensor_desc_exp *ops[3];
  pocl_dbk_unpack_bin_operands (kernel_id, kernel_attributes, ops);

  cl_int errcode;
  POCL_RETURN_ERROR_ON (ops[2]->rank < 1 || ops[2]->rank > 3,
                        CL_INVALID_TENSOR_RANK_EXP,
                        "Output's rank must be in range [1, 3]!");

  cl_tensor_shape_exp input_shape[4];
  unsigned input_rank
    = get_shape_for_implicit_broadcast (ops[0], ops[1], input_shape);

  POCL_RETURN_ERROR_ON (!input_rank, CL_INVALID_TENSOR_SHAPE_EXP,
                        "Shapes of the input operands don't match (after "
                        "implicit broadcasting)!");

  POCL_RETURN_ERROR_ON (input_rank != ops[2]->rank, CL_INVALID_TENSOR_RANK_EXP,
                        "Rank of the input and output operands don't match!");

  for (unsigned i = 0; i < input_rank; i++)
    {
      POCL_RETURN_ERROR_ON (
        input_shape[i] != ops[2]->shape[i], CL_INVALID_TENSOR_SHAPE_EXP,
        "Shape mismatch between input and output operands at dimension %u\n",
        i);
    }

  POCL_RETURN_ERROR_ON (
    (ops[0]->dtype != ops[1]->dtype || ops[1]->dtype != ops[2]->dtype),
    CL_INVALID_TENSOR_DATATYPE_EXP,
    "All operands must have the same element type!");

  return CL_SUCCESS;
}

int
pocl_release_eltwise_binary_dbk_attrs (cl_dbk_id_exp kernel_id,
                                       void *kernel_attributes)
{
  const cl_tensor_desc_exp *ops[3];
  pocl_dbk_unpack_bin_operands (kernel_id, kernel_attributes, ops);

  for (unsigned i = 0; i < 3; i++)
    pocl_tensor_destroy_body ((cl_tensor_desc_exp *)ops[0]);
  POCL_MEM_FREE (kernel_attributes);
  return CL_SUCCESS;
}

static void *
alloc_bin_dbk (cl_dbk_id_exp kernel_id, cl_tensor_desc_exp *new_ops[3])
{

#define HANDLE_CASE(_DBK_ID, _DBK_STRUCT)                                     \
  case _DBK_ID:                                                               \
    {                                                                         \
      _DBK_STRUCT *new_dbk = calloc (1, sizeof (_DBK_STRUCT));                \
      if (!new_dbk)                                                           \
        return NULL;                                                          \
      pocl_dbk_unpack_bin_operands (_DBK_ID, new_dbk,                         \
                                    (const cl_tensor_desc_exp **)new_ops);    \
      return new_dbk;                                                         \
    }

  switch (kernel_id)
    {
    default:
      assert (!"Unknown DBK!");
      return NULL;

      HANDLE_CASE (CL_DBK_ADD_EXP, cl_dbk_attributes_add_exp);
      HANDLE_CASE (CL_DBK_MUL_EXP, cl_dbk_attributes_mul_exp);
    }

#undef HANDLE_CASE
}

void *
pocl_copy_eltwise_binary_dbk_attrs (cl_dbk_id_exp kernel_id,
                                    const void *kernel_attributes)
{

  const cl_tensor_desc_exp *orig_ops[3];
  pocl_dbk_unpack_bin_operands (kernel_id, kernel_attributes, orig_ops);

  cl_tensor_desc_exp *new_ops[3];
  void *copy = alloc_bin_dbk (kernel_id, new_ops);
  if (!copy)
    return NULL;

  for (unsigned i = 0; i < 3; i++)
    if (pocl_tensor_copy (new_ops[i], orig_ops[i]))
      goto ERROR;

  return copy;

ERROR:
  pocl_release_eltwise_binary_dbk_attrs (kernel_id, copy);
  return NULL;
}
