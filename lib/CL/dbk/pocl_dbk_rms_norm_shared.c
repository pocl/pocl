/* pocl_dbk_rms_norm_shared.c - RMS norm DBK

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

#ifndef POCL_DBK_RMS_NORM_SHARED_H
#define POCL_DBK_RMS_NORM_SHARED_H

#include "pocl_dbk_rms_norm_shared.h"

#include "pocl_dbk_util.h"
#include "pocl_tensor_util.h"

int
pocl_validate_rms_norm_attrs (cl_dbk_id_exp kernel_id,
                              const void *kernel_attributes)
{
  const cl_dbk_attributes_rms_norm_exp *attrs = kernel_attributes;

  cl_int errcode;
  POCL_RETURN_ERROR_ON (attrs->src.rank < 1 || attrs->src.rank > 4,
                        CL_INVALID_TENSOR_RANK_EXP,
                        "Output's rank must be in range [1, 4]!");

  POCL_RETURN_ERROR_ON (!pocl_tensor_shape_equals (&attrs->src, &attrs->dst),
                        CL_INVALID_TENSOR_SHAPE_EXP,
                        "Shapes of the input operands don't match (after "
                        "implicit broadcasting)!");

  POCL_RETURN_ERROR_ON ((attrs->src.dtype != attrs->dst.dtype),
                        CL_INVALID_TENSOR_DATATYPE_EXP,
                        "All operands must have the same element type!");

  POCL_RETURN_ERROR_ON (attrs->start_dim >= attrs->src.rank,
                        CL_DBK_INVALID_ATTRIBUTE_EXP,
                        "Constraint violated: 'start_dim < src.rank'.");

  return CL_SUCCESS;
}

int
pocl_release_rms_norm_attrs (cl_dbk_id_exp kernel_id, void *kernel_attributes)
{

  cl_dbk_attributes_rms_norm_exp *attrs = kernel_attributes;
  pocl_tensor_destroy_body (&attrs->src);
  pocl_tensor_destroy_body (&attrs->dst);
  POCL_MEM_FREE (kernel_attributes);
  return CL_SUCCESS;
}

void *
pocl_copy_rms_norm_attrs (cl_dbk_id_exp kernel_id,
                          const void *kernel_attributes)
{
  const cl_dbk_attributes_rms_norm_exp *original = kernel_attributes;
  cl_dbk_attributes_rms_norm_exp *copy = calloc (1, sizeof (*original));
  if (!copy)
    return NULL;

  if (pocl_tensor_copy (&copy->src, &original->src))
    goto ERROR;

  if (pocl_tensor_copy (&copy->dst, &original->dst))
    goto ERROR;

  copy->start_dim = original->start_dim;
  memcpy (&copy->epsilon, &original->epsilon, sizeof (original->epsilon));

  return copy;

ERROR:
  pocl_release_rms_norm_attrs (kernel_id, copy);
  return NULL;
}

#endif // POCL_DBK_RMS_NORM_SHARED_H
